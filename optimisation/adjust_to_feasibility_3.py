import pandas as pd
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, Set, Param, 
                           NonNegativeReals, Binary, SolverFactory, minimize, value)
from optimisation.baseline_model import setup_model
import time
import copy
from operator import itemgetter


def adjust_to_feasibility_3(model, epsilon=1e-6, max_iterations=100):
    start = time.time()
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

        # Step 2: Redistribute deficits to meet remaining demands using cost-based allocation
        for d in model.D:
            for t in model.T:
                total_demand = value(model.Demand[d, t])
                supplied = sum(adjusted_x.get((s, d, t), 0) for s in model.S)
                deficit = total_demand - supplied

                if deficit > epsilon:
                    violations_found = True

                    # Create a list of supply sites sorted by supply cost (ascending)
                    supply_sites_sorted = sorted(
                        model.S,
                        key=lambda s: value(model.SupplyCost[s, d, t])
                    )

                    # Fill deficit starting with the cheapest supply sites
                    for s in supply_sites_sorted:
                        total_supply_current_st = sum(adjusted_x.get((s, d_, t), 0) for d_ in model.D)
                        remaining_capacity = value(model.MaxCapacity[s]) - total_supply_current_st

                        if remaining_capacity > epsilon:
                            additional_supply = min(deficit, remaining_capacity)
                            original_supply = adjusted_x.get((s, d, t), 0)
                            adjusted_x[(s, d, t)] = original_supply + additional_supply
                            deficit -= additional_supply

                            # Optional: Debugging statement
                            # print(f"Allocated additional supply of {additional_supply:.4f} from Supply Site {s} to Demand Site {d} at Time {t}")

                        if deficit <= epsilon:
                            break

        if not violations_found:
            converged = True
            print("No capacity violations or deficits detected. Adjustment converged.")
        else:
            print(f"Iteration {iteration}: Violations detected. Proceeding to next iteration.")

    if iteration == max_iterations and not converged:
        print("Maximum adjustment iterations reached. Solution may still have violations.")

    # Step 3: Determine which sites are constructed based on adjusted_x
    for s in model.S:
        supplied_from_s = sum(adjusted_x.get((s, d, t), 0) for d in model.D for t in model.T)
        feasible_y[s] = 1 if supplied_from_s > epsilon else 0
        if feasible_y[s] == 1:
            constructed_sites.add(s)

    # Step 4: Calculate implied feasible cost
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
            assert supplied >= total_demand - epsilon, f"Demand not met for Demand Site {d}, Time {t}. Supplied: {supplied:.4f}, Required: {total_demand:.4f}"

    for s in model.S:
        for t in model.T:
            total_supply = sum(adjusted_x.get((s, d, t), 0) for d in model.D)
            max_cap = value(model.MaxCapacity[s])
            assert total_supply <= max_cap + epsilon, f"Capacity exceeded for Supply Site {s}, Time {t}. Supplied: {total_supply:.4f}, Max: {max_cap:.4f}"
    print("This adjustment took: ", time.time() - start)
    return adjusted_x, feasible_y, feasible_cost
