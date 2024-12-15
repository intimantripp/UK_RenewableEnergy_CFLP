import pandas as pd
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, Set, Param,
                           NonNegativeReals, Binary, SolverFactory, minimize, value)
from optimisation.baseline_model import setup_model


def adjust_to_feasibility(model):
    """
    Adjust the LR solution to a feasible solution for the original problem by scaling
    down x[s, d, t] proportionally to fit capacity constraints
    """
    adjusted_x = {}
    feasible_cost = 0

    for s in model.S:
        for t in model.T:
            total_supply = sum(value(model.x[s, d, t]) for d in model.D)
            max_capacity = value(model.y[s]) * value(model.MaxCapacity[s])

            if total_supply > max_capacity:
                # Scale down supply proportionally
                scale = max_capacity / total_supply if total_supply > 0 else 1
                for d in model.D:
                    adjusted_x[(s, d, t)] = value(model.x[s, d, t]) * scale
            else:
                for d in model.D:
                    adjusted_x[(s, d, t)] = value(model.x[s, d, t])

    # Compute the feasible solution cost
    for (s, d, t), x_val in adjusted_x.items():
        feasible_cost += model.SupplyCost[s, d, t] * x_val

    return adjusted_x, feasible_cost

def setup_model_LR(demand_df, supply_df, costs_df, lambda_values):

    # Define model
    model = ConcreteModel()

    # Sets
    model.S = Set(initialize=supply_df['Supply Site'].unique())
    model.D = Set(initialize=demand_df['Demand Site'].unique())
    model.T = Set(initialize=demand_df['Year'].unique())

    # Define parameters
    setup_costs = supply_df.set_index('Supply Site')['Setup Costs'].to_dict()
    model.SetupCost = Param(model.S, initialize=setup_costs)

    demand_data = demand_df.set_index(['Demand Site', 'Year'])['Demand'].to_dict()
    model.Demand = Param(model.D, model.T, initialize=demand_data)

    # Energy supply costs
    supply_costs = costs_df.set_index(['Supply Site', 'Demand Site', 'Year'])['Supply Cost'].to_dict()
    model.SupplyCost = Param(model.S, model.D, model.T, initialize=supply_costs)

    #Maximum Capacity parameter
    max_capacity = supply_df.set_index(['Supply Site'])['Capacity'].to_dict()
    model.MaxCapacity = Param(model.S, initialize=max_capacity)

    # Lagrange multiplies as given parameters
    model.Lambda = Param(model.S, model.T, initialize=lambda_values)

    # Decision variables
    model.y = Var(model.S, domain=Binary) # y determines whether a supply site is active
    model.x = Var(model.S, model.D, model.T, domain=NonNegativeReals) # x determines amount supplied at time t by one site to another

    def objective_rule(model):
        setup_cost = sum(model.SetupCost[s]*model.y[s] for s in model.S)
        supply_cost = sum(model.SupplyCost[s, d, t] * model.x[s, d, t]
                        for s in model.S for d in model.D for t in model.T)
        
        lagrangian_term = sum(
            model.Lambda[s, t] * (sum(model.x[s, d, t] for d in model.D) - model.y[s]*model.MaxCapacity[s])
            for s in model.S for t in model.T
            )

        return setup_cost + supply_cost + lagrangian_term

    model.Objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints

    # demand satisfaction
    def demand_satisfaction_rule(model, d, t):
        return sum(model.x[s, d, t] for s in model.S) >= model.Demand[d, t]
    model.DemandSatisfaction = Constraint(model.D, model.T, rule=demand_satisfaction_rule)

    return model

def create_results_dfs(results, model, lambda_values, iterations, true_cost):

    supply_results = []
    for s in model.S:
        for d in model.D:
            for t in model.T:
                energy_provided = value(model.x[s, d, t])
                if energy_provided > 0:
                    supply_results.append({
                    'Supply Site': s,
                    'Demand Site': d,
                    'Year': t,
                    'Energy Provided (MW)': energy_provided
                    })
    supply_results_df = pd.DataFrame(supply_results)

    constructed_sites = [s for s in model.S if model.y[s].value == 1]
    solve_time = results.solver.time
    summary_df = pd.DataFrame({
        "Solve Time (seconds)": [solve_time],
        "True Cost": [true_cost],
        "Final Objective Value": [value(model.Objective)],
        "Number of Iterations": [iterations],
        "Constructed Sites": [constructed_sites],
        "Lambda Values": [lambda_values],
    })

    return supply_results_df, summary_df

def run_model_LR_with_heuristic(demand_df, supply_df, costs_df, output_directory="", output_suffix="",
                                model_solver="gurobi", save_results=True, max_iterations=50, 
                                step_size=100, tolerance=1e-6):
    """
    Solve the Lagrangian Relaxation problem and use a heuristic to adjust to a feasible solution.
    """
    # Preprocess data
    supply_df['Supply Site'] = supply_df['Supply Site'].str.strip()
    demand_df['Demand Site'] = demand_df['Demand Site'].str.strip()
    costs_df['Supply Site'] = costs_df['Supply Site'].str.strip()
    costs_df['Demand Site'] = costs_df['Demand Site'].str.strip()

    # Initialize lambda values

    lambda_values = {}
    for s in supply_df['Supply Site'].unique():
        for t in demand_df['Year'].unique():
            # Initialise with slightly positive values, initialising at 0 causes issues 
            lambda_values[(s, t)] = 0.0

    solver = SolverFactory(model_solver)

    # Initialise bounds
    Z_LR = float('-inf')
    Z_feas = float('inf')

    for iteration in range(max_iterations):
        
        # Solve the LR problem
        model = setup_model_LR(demand_df, supply_df, costs_df, lambda_values)
        results = solver.solve(model, tee=False)

        # Compute the relaxed problem cost (Z_LR)
        Z_LR_iter = value(model.Objective)
        Z_LR = max(Z_LR, Z_LR_iter)

        # Adjust to a feasible solution and compute its cost (Z_feas)
        adjusted_x, feasible_cost = adjust_to_feasibility(model)
        Z_feas = max(Z_LR, Z_LR_iter) # Update the lower bound

        # Check stopping criterion
        if (Z_feas - Z_LR) / Z_feas < tolerance:
            break

        # Compute capacity violations
        violations = {}
        for s in model.S:
            for t in model.T:
                total_supply = sum(value(model.x[s, d, t]) for d in model.D)
                max_capacity = value(model.y[s]) * value(model.MaxCapacity[s])
                violation = total_supply - max_capacity
                violations[(s, t)] = violation

        # Update lambda values using subgradient method
        for (s, t), v in violations.items():
            if v > 0:
                lambda_values[(s, t)] += step_size * v
            lambda_values[(s, t)] = max(0, lambda_values[(s, t)]) # Ensure non-negative multipliers

        # Dynamically decrease step size for stability
        step_size = step_size / (1 + iteration)

    # Obtain cost of solution found by LR Heuristic
    baseline_model = setup_model(demand_df, supply_df, costs_df)

    # Fix variables using LR solution and adjusted x vals
    for s in baseline_model.S:
        baseline_model.y[s].fix(value(model.y[s]))
    for (s, d, t), x_val in adjusted_x.items():
        baseline_model.x[s, d, t].fix(x_val)
    
    # Evaluate true cost
    true_cost = value(baseline_model.Objective)

    # Output results
    supply_results, summary_results = create_results_dfs(results, model, lambda_values, iteration, true_cost)

    # Add heuristic results to summary
    summary_results['Lower Bound (Z_LR)'] = Z_LR
    summary_results['Upper Bound (Z_feas)'] = Z_feas

    if save_results:
        supply_results.to_csv(f"{output_directory}supply_results{output_suffix}.csv",
                               index=False)
        summary_results.to_csv(f"{output_directory}summary_results{output_suffix}.csv",
                               index=False)

    return supply_results, summary_results

                                     
