import pandas as pd
from optimisation.baseline_model import run_baseline_model
from optimisation.lr_heuristic import run_model_LR_with_heuristic
from utils.feasibility_check import check_feasibility
from utils.rename_cols import rename_headers
from optimisation.lr_heurstic_2 import run_LR_heuristic

parameters = {
    "solver": "gurobi",
    "max_iterations": 50,
    "step_size": 1e5,
    "tolerance": 0.1,
    "output_directory": "Outputs/",
    "run_baseline": True,
    "run_LR_heuristic": True,
    "baseline_suffix": "test_5",
    "lr_suffix": "test_5",
    "Big M": 0
}

# Load Data
# Dummy Test 2
# supply_df = pd.read_csv("Data/dummy_test/dummy_supply.csv")
# demand_df = pd.read_csv("Data/dummy_test/dummy_demand.csv")
# costs_df = pd.read_csv("Data/dummy_test/dummy_costs_2.csv")


# # Test 1
# supply_df = pd.read_csv("Data/test_1_supply_n5.csv")
# demand_df = pd.read_csv("Data/test_1_demand_n3.csv")
# costs_df = pd.read_csv("Data/test_1_costs_n3_n5.csv")


# # Test 2
# supply_df = pd.read_csv("Data/test_2_supply_n10.csv")
# demand_df = pd.read_csv("Data/test_2_demand_n10.csv")
# costs_df = pd.read_csv("Data/test_2_costs_n10_n10.csv")


# Test 3
# supply_df = pd.read_csv("Data/test_3_supply_n30.csv")
# demand_df = pd.read_csv("Data/test_3_demand_n30.csv")
# costs_df = pd.read_csv("Data/test_3_costs_n30_n30.csv")


# Test 4
# supply_df = pd.read_csv("Data/test_4_supply_n50.csv")
# demand_df = pd.read_csv("Data/test_4_demand_n50.csv")
# costs_df = pd.read_csv("Data/test_4_costs_n50_n50.csv")


# Test 5
supply_df = pd.read_csv("Data/test_5_supply_n100.csv")
demand_df = pd.read_csv("Data/test_5_demand_n100.csv")
costs_df = pd.read_csv("Data/test_5_costs_n100_n100.csv")





supply_df, demand_df, costs_df = rename_headers(supply_df, demand_df, costs_df, 
                                                supply_col_dict={'Annual Capacity (GW)': 'Capacity',
                                                                 "Setup Cost": "Setup Costs"},
                                                demand_col_dict={'Annual Demand (GWh)': 'Demand'},
                                                costs_df_dict={}
                                                )


if __name__ == "__main__":

    is_feasible = check_feasibility(demand_df, supply_df)
    if not is_feasible:
        print("Problem isn't feasible. Examnine the datasets")
    print(supply_df.columns)
    if parameters['run_baseline'] and is_feasible:
        print("Runing baseline model...")
        supply_results_baseline, _ = run_baseline_model(
            demand_df, supply_df, costs_df, 
            output_directory=parameters['output_directory'],
            output_suffix=f"_baseline_{parameters['baseline_suffix']}"
            )
        print("Baseline run complete")
    
    if parameters['run_LR_heuristic'] and is_feasible:
        print("Running LR heuristic")
        supply_results_LR, _ = run_LR_heuristic(demand_df, supply_df, costs_df, step_size=parameters['step_size'], 
                         output_directory=parameters['output_directory'],
                         output_suffix=f"_LR_{parameters['lr_suffix']}",
                         max_iterations=parameters['max_iterations'], tolerance=parameters['tolerance'], big_M=parameters['Big M'])
        print("LR Heuristic run complete")
    if supply_results_baseline is not None and supply_results_LR is not  None:
        print(f"Output supplies match: {supply_results_LR.equals(supply_results_baseline)}")
    