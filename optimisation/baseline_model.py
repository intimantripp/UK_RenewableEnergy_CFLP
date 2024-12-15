import pandas as pd
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, Set, Param, 
                           NonNegativeReals, Binary, SolverFactory, minimize, value)

def setup_model(demand_df, supply_df, costs_df):

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

    # Decision variables
    model.y = Var(model.S, domain=Binary) # y determines whether a supply site is active
    model.x = Var(model.S, model.D, model.T, domain=NonNegativeReals) # x determines amount supplied at time t by one site to another

    def objective_rule(model):
        setup_cost = sum(model.SetupCost[s]*model.y[s] for s in model.S)
        supply_cost = sum(model.SupplyCost[s, d, t] * model.x[s, d, t]
                        for s in model.S for d in model.D for t in model.T)
        return setup_cost + supply_cost

    model.Objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints

    # demand satisfaction
    def demand_satisfaction_rule(model, d, t):
        return sum(model.x[s, d, t] for s in model.S) >= model.Demand[d, t]
    model.DemandSatisfaction = Constraint(model.D, model.T, rule=demand_satisfaction_rule)

    # Supply capacity
    def supply_capacity_rule(model, s, t):
        return sum(model.x[s, d, t] for d in model.D) <= model.y[s] * model.MaxCapacity[s]
    model.SupplyCapacity = Constraint(model.S, model.T, rule=supply_capacity_rule)

    return model

def create_results_dfs(results, model):

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
                    'Energy Provided': energy_provided
                    })
    supply_results_df = pd.DataFrame(supply_results)

    constructed_sites = [s for s in model.S if model.y[s].value == 1]
    solve_time = results.solver.time
    summary_df = pd.DataFrame({
        "Solve Time (seconds)": [solve_time],
        "Final Objective Value": [value(model.Objective)],
        "Constructed Sites": [constructed_sites],
    })

    return supply_results_df, summary_df


def run_baseline_model(demand_df, supply_df, costs_df, output_directory="", output_suffix="", model_solver='gurobi',
              save_results=True):
    
    # Preprocess data
    supply_df['Supply Site'] = supply_df['Supply Site'].str.strip()
    demand_df['Demand Site'] = demand_df['Demand Site'].str.strip()
    costs_df['Supply Site'] = costs_df['Supply Site'].str.strip()
    costs_df['Demand Site'] = costs_df['Demand Site'].str.strip()

    #create the model
    model = setup_model(demand_df, supply_df, costs_df)

    #create the solver
    solver = SolverFactory(model_solver)
    results = solver.solve(model)

    supply_results, summary_results = create_results_dfs(results, model)
    
    if save_results:
        supply_results.to_csv(f"{output_directory}supply_results{output_suffix}.csv",
                               index=False)
        summary_results.to_csv(f"{output_directory}summary_results{output_suffix}.csv",
                               index=False)
    return supply_results, summary_results
