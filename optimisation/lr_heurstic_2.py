import pandas as pd
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, Set, Param, 
                           NonNegativeReals, Binary, SolverFactory, minimize, value)
from optimisation.baseline_model import setup_model
import time


def setup_model_LR(demand_df, supply_df, costs_df, lambda_values, M):

    model = ConcreteModel()

    # Define sets
    model.S = Set(initialize=supply_df['Supply Site'].unique())
    model.D = Set(initialize=demand_df['Demand Site'].unique())
    model.T = Set(initialize=demand_df['Year'].unique())

    # Parameters
    setup_costs = supply_df.set_index("Supply Site")['Setup Costs'].to_dict()
    model.SetupCost = Param(model.S, initialize=setup_costs)
    
    demand_data = demand_df.set_index(['Demand Site', 'Year'])['Demand'].to_dict()
    model.Demand = Param(model.D, model.T, initialize=demand_data)

    supply_costs = costs_df.set_index(['Supply Site', 'Demand Site', 'Year'])['Supply Cost'].to_dict()
    model.SupplyCost = Param(model.S, model.D, model.T, initialize=supply_costs)

    max_capacity = supply_df.set_index(['Supply Site'])['Capacity'].to_dict()
    model.MaxCapacity = Param(model.S, initialize=max_capacity)

    # Lagrange multipliers
    model.Lambda = Param(model.S, model.T, initialize=lambda_values)

    model.y = Var(model.S, domain=Binary)
    model.x = Var(model.S, model.D, model.T, domain=NonNegativeReals)

    # Objective: original + penalty for violating capacity constraint
    def objective_rule(model):
        setup_term = sum(model.SetupCost[s]*model.y[s] for s in model.S)
        supply_term = sum(model.SupplyCost[s, d, t]*model.x[s, d, t] for s in model.S for d in model.D for t in model.T)
        lagrangian_term = sum(
            model.Lambda[s, t] * (sum(model.x[s, d, t] for d in model.D) - model.y[s]*model.MaxCapacity[s])
            for s in model.S for t in model.T
        )
        penalty_term = sum(M*model.x[s, d, t]*(1 - model.y[s]) 
                        for s in model.S for d in model.D for t in model.T)
        return setup_term + supply_term + lagrangian_term + penalty_term
    
    model.Objective = Objective(rule=objective_rule, sense=minimize)


    # Demand satisfaction - total supply to demand site d at t must be at least the 
    # demand d has at t
    def demand_satisfaction_rule(model, d, t):
        return sum(model.x[s, d, t] for s in model.S) >= model.Demand[d, t]
    
    model.DemandSatisfaction = Constraint(model.D, model.T, rule=demand_satisfaction_rule)

    return model


def adjust_to_feasibility_2(model, epsilon=1e-6, max_iterations=100):

    import copy
    adjusted_x = {}
    feasible_y = {}
    constructed_sites = set()
    iteration = 0
    converged = False

    # Initialise adjusted_x with the current LR solution
    for s in model.S:
        for d in model.D:
            for t in model.T:
                adjusted_x[(s, d, t)] = value(model.x[s, d, t])

    while not converged and iteration < max_iterations:
        iteration += 1
        # print("Iteration", iteration)
        violations_found = False

        # Step 1: Scale down s[s, d, t] to align with capacity constraints
        for s in model.S:
            for t in model.T:
                total_supply = sum(adjusted_x.get((s, d, t), 0) for d in model.D)
                max_capacity = value(model.MaxCapacity[s])
                
                if total_supply > max_capacity + epsilon and max_capacity > 0:
                    # print("Capacity exceeded")
                    violations_found = True
                    scale = max_capacity / total_supply
                    for d in model.D:
                        original_supply = adjusted_x.get((s, d, t), 0)
                        scaled_supply = original_supply * scale
                        adjusted_x[(s, d, t)] = scaled_supply

        # Step 2: Redistribute deficits to meet remaining demands
        for d in model.D:
            for t in model.T:
                total_demand = value(model.Demand[d, t])
                supplied = sum(adjusted_x.get((s, d, t), 0) for s in model.S)
                deficit = total_demand - supplied
                if deficit > epsilon:
                    violations_found = True

                    # Fill deficit
                    for s in model.S:
                        total_supply_current_st = sum(adjusted_x.get((s, d_, t), 0) for d_ in model.D)
                        remaining_capacity = value(model.MaxCapacity[s]) - total_supply_current_st
                        if remaining_capacity > epsilon:
                            additional_supply = min(deficit, remaining_capacity)
                            original_supply = adjusted_x.get((s, d, t), 0)
                            adjusted_x[s, d, t] = original_supply + additional_supply
                            deficit -= additional_supply

                        # print(f"Allocated addiitonal supply")
                        if deficit <= epsilon:
                            break

        if not violations_found:
            converged = True
            print("No capacity violations or deficits detected. Adjustment converged.")
        else:
            print("Violations detected. Proceeding to next iteration")
    if iteration == max_iterations and not converged:
        print("Maximum adjustment iterations reached. Solution may still have violations")
    
    # Step 3: Determine which sites are constructed based on adjusted_x
    for s in model.S:
        supplied_from_s = sum(adjusted_x.get((s, d, t), 0) for d in model.D for t in model.T)
        feasible_y[s] = 1 if supplied_from_s > epsilon else 0
        if feasible_y[s] == 1:
            constructed_sites.add(s)
        # if feasible_y[s] == 1 and value(model.y[s]) != 1:
        #     print(f"Site {s} provides {supplied_from_s:.4f} energy but its y is {value(model.y[s]):.4f}. Setting y[s] to 1.")
    
    # Step 4: calculate implied feasible cost
    feasible_cost = 0
    for s in model.S:
        if feasible_y[s] == 1:
            feasible_cost += value(model.SetupCost[s])
    for (s, d, t), x_val in adjusted_x.items():
        feasible_cost += model.SupplyCost[s, d, t] * x_val
    
    # Step 5: Validation Checks
    for d in model.D:
        for t in model.T:
            total_demand = value(model.Demand[d, t])
            supplied = sum(adjusted_x.get((s, d, t), 0) for s in model.S)
            assert supplied >= total_demand - epsilon, f"Demand not met for Demand Site {d}, Year {t}. Supplied: {supplied:.4f}, Required: {total_demand:.4f}"
    
    for s in model.S:
        for t in model.T:
            total_supply = sum(adjusted_x.get((s, d, t), 0) for d in model.D)
            max_cap = value(model.MaxCapacity[s])
            assert total_supply <= max_cap + epsilon, f"Capacity exceeded for Supply Site {s}, Year {t}. Supplied: {total_supply:.4f}, Max: {max_cap:.4f}"
    
    return adjusted_x, feasible_y, feasible_cost
        


def adjust_to_feasbility(model):
    """
    Given an LR solution (x, y), produce a feasible solution for the original problem 
    ensuring capacity capacity constraints are met. This means scaling down x if capacity is exceeded,
    ensuring y is 1 if s is supplying anything
    """


    adjusted_x = {}

    # First ensure capacity not violated
    for s in model.S:
        for t in model.T:
            total_supply = sum(value(model.x[s, d, t]) for d in model.D)
            max_capacity = value(model.MaxCapacity[s]) #* value(model.y[s]) 
            print(total_supply, max_capacity)
            if total_supply > max_capacity and max_capacity > 0:
                print("Site supply exceeds capacity")
                scale = max_capacity/total_supply
                for d in model.D:
                    adjusted_x[(s, d, t)] = value(model.x[s, d, t]) * scale
            else:
                for d in model.D:
                    adjusted_x[(s, d, t)] = value(model.x[s, d, t])
    
    # Redistribute deficits to meet demands
    for d in model.D:
        for t in model.T:
            total_demand = value(model.Demand[d, t])
            supplied = sum(adjusted_x.get((s, d, t), 0) for s in model.S)
            deficit = total_demand - supplied
            if deficit > 1e-6:
                print(f"Deficit detected for demand site {d}, year {t}: {deficit}")
                for s in model.S:
                    # Calculate remaining capacity for site s in year t
                    total_supply_current_st = sum(adjusted_x.get((s, d_, t), 0) for d_ in model.D)
                    remaining_capacity = value(model.MaxCapacity[s]) - total_supply_current_st
                    if remaining_capacity > 1e-6:
                        additional_supply = min(deficit, remaining_capacity)
                        adjusted_x[(s, d, t)] = adjusted_x.get((s, d, t), 0) + additional_supply
                        deficit -= additional_supply
                        # print(f"Allocated additional {additional_supply} to Site {s}, Year {t}")
                        if deficit <= 1e-6:
                            break

    # Ensure that y's are consistent. If s supplies energy, y[s] must equal 1
    feasible_y = {}
    for s in model.S:
        supplied_from_s = sum(adjusted_x.get((s, d, t), 0) for d in model.D for t in model.T)
        # if supplied_from_s > 1e-6 and value(model.y[s]) != 1:
        #     print(f"Site {s} provides {supplied_from_s} energy but its y is {value(model.y[s])}")
        feasible_y[s] = 1 if supplied_from_s > 1e-6 else 0
    
    # Calculate the implied feasible cost
    feasible_cost = 0
    for s in model.S:
        if feasible_y[s] == 1:
            feasible_cost += value(model.SetupCost[s])
    for (s, d, t), x_val in adjusted_x.items():
        feasible_cost += model.SupplyCost[s, d, t] * x_val

    # Validation Checks
    for d in model.D:
        for t in model.T:
            total_demand = value(model.Demand[d, t])
            supplied = sum(adjusted_x.get((s, d, t), 0) for s in model.S)
            assert supplied >= total_demand - 1e-6, f"Demand not met for Demand Site {d}, Year {t}"

    for s in model.S:
        for t in model.T:
            total_supply = sum(adjusted_x.get((s, d, t), 0) for d in model.D)
            max_cap = value(model.MaxCapacity[s])
            assert total_supply <= max_cap, f"Capacity exceeded for Supply Site {s}, Year {t}"

    
    return adjusted_x, feasible_y, feasible_cost


def evaluate_solution_in_baseline(demand_df, supply_df, _costs_df, feasible_x, feasible_y, solver='gurobi'):

    baseline = setup_model(demand_df, supply_df, _costs_df)
    for s in baseline.S:
        baseline.y[s].fix(feasible_y[s])
    for s in baseline.S:
        for d in baseline.D:
            for t in baseline.T:
                baseline.x[s, d, t].fix(feasible_x.get((s, d, t), 0))

    solver = SolverFactory(solver)
    res = solver.solve(baseline, tee=False) #Solution should be feasible
    true_cost = value(baseline.Objective)
    return true_cost
                                  

def create_LR_results_dfs(model, adjusted_x, feasible_y, true_cost, time_taken,
                          iteration, final_gap):
    
    supply_results = []
    for s in model.S:
        for d in model.D:
            for t in model.T:
                energy_provided = adjusted_x[(s, d, t)]
                if energy_provided > 0:
                    supply_results.append({
                        'Supply Site': s,
                        'Demand Site': d,
                        'Year': t,
                        'Energy Provided': energy_provided
                    })
    supply_results_df = pd.DataFrame(supply_results)

    constructed_sites = [s for s in model.S if feasible_y[s] == 1]
    summary_df = pd.DataFrame({
        "Solve Time (seconds)": [time_taken],
        "Final Objective Vale": [true_cost],
        "Number of iterations": [iteration],
        "Final Gap": [final_gap],
        "Constructed Sites": [constructed_sites]
    })

    return supply_results_df, summary_df




def run_LR_heuristic(demand_df, supply_df, costs_df, model_solver='gurobi',
                    max_iterations=50, tolerance=1e-4, step_size=10, save_results=True,
                    output_directory="", output_suffix="", big_M=0):
    """
    Run the LR heuristic
    1. Initialise lambda values
    2. Solve the LR problem
    3. Use LR solution as lower bound to problem (Z_LR)
    4. Adjust solution to feasibility, obtain upper bound (Z_feas)
    5. Check gap between Z_feas - Z_LR - if small have obtained solution - stop
    6. Update lambdas based on violations in LR solution
    7. Go to step 2 using updated lambdas
    """


    # Preprocess data
    supply_df['Supply Site'] = supply_df['Supply Site'].str.strip()
    demand_df['Demand Site'] = demand_df['Demand Site'].str.strip()
    costs_df['Supply Site'] = costs_df['Supply Site'].str.strip()
    costs_df['Demand Site'] = costs_df['Demand Site'].str.strip()

    solver = SolverFactory(model_solver)

    # Initialise lambda values
    lambda_values = {}
    for s in supply_df['Supply Site'].unique():
        for t in demand_df['Year'].unique():
            lambda_values[(s, t)] = 0.0 # small positive multiplier initially. 0s lead to poor solutions
    
    Z_LR = float('-inf') # initial lower bound
    Z_feas = float('inf') # initial upper bound
    previous_Z_LR = None

    start_time = time.time()

    for iteration in range(max_iterations):

        print(iteration)
        # Solve the LR problem
        lr_model = setup_model_LR(demand_df, supply_df, costs_df, lambda_values, big_M)
        lr_results = solver.solve(lr_model, tee=False)

        # Use LR solution to update lower bound
        lr_obj = value(lr_model.Objective)
        Z_LR = max(Z_LR, lr_obj)


        # Adjust LR solution to feasible solution
        adjusted_x, feasible_y, feasible_cost = adjust_to_feasibility_2(lr_model)
        Z_feas = min(Z_feas, feasible_cost)
        print("Z_LR: ", Z_LR, "Z_feas: ", Z_feas)
        if Z_feas > 0:
            gap = (Z_feas - Z_LR)/Z_feas

        else: gap = float('inf')
        print("gap: ", gap)
        if gap < tolerance:
            # We have found a solution
            # Use x_adjusted and y_feasible as solutions
            break
        
        # Calculate violations and update corresponding lambdas
        violations = {}
        for s in lr_model.S:
            for t in lr_model.T:
                total_supply = sum(value(lr_model.x[s, d, t]) for d in lr_model.D)
                max_capacity = value(lr_model.y[s] * value(lr_model.MaxCapacity[s]))
                violation = total_supply - max_capacity
                violations[(s, t)] = violation
        
        if all(v <= 0 for v in violations.values()):
        # No positive violations
        # Check if LR objective is stable
            if previous_Z_LR is not None and abs(Z_LR - previous_Z_LR) < 1e-9:
                print("No violations and LR objective stable. Stopping.")
                break

        for (s, t), v in violations.items():
            if v > 0:
                # print("We have a violation: ",(s, t))
                lambda_values[(s, t)] += step_size * v
            lambda_values[(s, t)] = max(0, lambda_values[(s, t)])

        # Decrease step size
        step_size *= 0.999
        # Record Z_LR 
        previous_Z_LR = Z_LR

    end_time = time.time()
    time_taken = end_time - start_time

    true_cost = evaluate_solution_in_baseline(demand_df, supply_df, costs_df, adjusted_x, feasible_y)

    supply_results, summary_results = create_LR_results_dfs(lr_model, adjusted_x, feasible_y,
                                                         true_cost, time_taken, iteration, gap)

    if save_results:
        supply_results.to_csv(f"{output_directory}supply_results{output_suffix}.csv",
                               index=False)
        summary_results.to_csv(f"{output_directory}summary_results{output_suffix}.csv",
                               index=False)
    return supply_results, summary_results


