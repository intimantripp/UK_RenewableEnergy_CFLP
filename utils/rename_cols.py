import pandas as pd

def rename_headers(supply_df, demand_df, costs_df, supply_col_dict={}, demand_col_dict={}, costs_df_dict={}):
    supply_df = supply_df.rename(columns=supply_col_dict)
    demand_df = demand_df.rename(columns=demand_col_dict)
    costs_df = costs_df.rename(columns=demand_col_dict)

    return supply_df, demand_df, costs_df