import pandas as pd

def check_feasibility(demand_df, supply_df):
    """
    Checks that the annual energy demanded for each year is always less than or equal to
    the maximum annual output that can be provided from the supply df
    """

    max_annual_demand = demand_df[['Year', 'Demand']].groupby('Year').sum().max().iloc[0]
    total_supply_capacity = supply_df['Capacity'].sum()
    is_feasible = max_annual_demand <= total_supply_capacity
    return is_feasible
