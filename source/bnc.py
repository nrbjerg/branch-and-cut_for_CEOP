from source.data_types import Matrix, Vector
from typing import Tuple, Optional
import gurobipy as gp
import numpy as np
from itertools import product
from source.preprocessing import preprocess_CEOP, preprocess_SOP
from source.CE_path_TSP import solve_CE_path_TSP_under_ordering, solve_CE_path_TSP, strengthen_minimum_infeasible_vertex_set, extract_order_from_partial_CE_path_TSP_order
from source.helpers import load_instance, plot_solution, PATH_TO_INSTANCES, find_subtours
from source.analysis import plot_incumbents_and_cover_cuts
from source.discretization import xi_IPD_scheme
from gurobipy import GRB
import os 
import time

cover_cut_times, incumbent_CEOP_solution_times, incumbent_CEOP_solution_scores, upper_bounds, upper_bounds_times, total_runtime_of_cut_generation = [], [], [], [], [], 0.0

def solve_BNC(p: Matrix, s: Vector, t_max: float, r: float, K: int, time_limit: Optional[float] = None) -> Tuple[float, Matrix, float]:
    """Solves the CEOP by discretizing it as a GOP using the IPD scheme and allocating a non-restrictive maximum distance budget,
       within this GOP and subsequently solving a CE path TSP using SOCP to check if the solution is a feasible solution for the CEOP.
       Otherwise we may cut the solution from the solution space, in a similar fashion as adding a sub-tour elimination constraint."""
    global cover_cut_times, incumbent_CEOP_solution_times, upper_bounds, upper_bounds_times, total_runtime_of_cut_generation
    cover_cut_times, incumbent_CEOP_solution_times, incumbent_CEOP_solution_scores, upper_bounds, upper_bounds_times, total_runtime_of_cut_generation = [], [], [], [], [], 0.0

    start_time = time.perf_counter()
    p, s, t_max, r = preprocess_CEOP(p, s, t_max, r)

    # Index stuff
    N = p.shape[0]
    M = 2 + (N - 2) * K
    idx = lambda n: ([0] if n == 0 else [M - 1]) if n == 0 or n == N - 1 else range(1 + (n - 1) * K, 1 + n * K) # NOTE: Corresponds to \mathcal{I} from the paper.
    
    _, d, adj = preprocess_SOP(p, s, t_max, r, K)
    end_time_preprocessing = time.perf_counter()

    arcs = [(i, j) for i, j in product(range(M), range(M)) if adj[i, j]]
    incoming = {j: [i for i in range(M - 1) if adj[i, j]] for j in range(1, M)}
    outgoing = {i: [j for j in range(1, M) if adj[i, j]] for i in range(M - 1)}
    arcs_between_nodes = {(n, n_prime): [(i, j) for (i, j) in product(idx(n), idx(n_prime)) if adj[i, j]] for n, n_prime in product(range(N - 1), range(1, N))}

    def strengthen_model(model: gp.Model):
        """Strengthens the model lightly by imposing extra valid constraints."""
        # Add exclusionary constraints based on arcs.
        for n in range(1, N - 1):
            exclusionary_nodes = [n_prime for n_prime in range(1, N - 1) if n_prime != n and len(arcs_between_nodes[(n, n_prime)]) == 0 and len(arcs_between_nodes[(n_prime, n)]) == 0]
            model.addConstrs((y[n] + y[n_prime] <= 1 for n_prime in exclusionary_nodes), name=f"Exclusionary nodes")

    # Stuff for keeping track of TSP results in callback.
    dists = dict()
    seen = set()
    
    def callback_with_SE(model: gp.Model, where):
        """Checks if the solution found is feasible, in which case we have found a feasible solution for the CEOP, if we can check if the CE path TSP
           solution is feasible within the CEOP framework i.e. does it obey the imposed maximum distance budget t_max? If not we may include a sub-tour 
           elimination style cut, we may also include such a cut if the solution is not connected etc."""
        global cover_cut_times, incumbent_CEOP_solution_times, upper_bounds, upper_bounds_times, total_runtime_of_cut_generation
        if model._time_limit is not None:
            elapsed = time.perf_counter() - model._start_time
            if elapsed >= model._time_limit:
                model.terminate()
                return

        if where == GRB.Callback.MIP and (upper_bound := model.cbGet(GRB.Callback.MIP_OBJBND)) <= upper_bounds[-1] - 1e-4: # NOTE: a weird artifact where the first upper bound is much greater than sum(s)
            upper_bounds.append(upper_bound) 
            upper_bounds_times.append(time.perf_counter() - end_time_preprocessing)
            
        if where == GRB.Callback.MIPSOL:
            # Log upper bounds 

            # Check if the solution is connected, otherwise add a cut for each sub-tour. 
            x_vals = model.cbGetSolution(x)
            selected_arcs = [(n, m) for n, m in product(range(N - 1), range(1, N)) if any([x_vals[i, j] >= 0.05 for (i, j) in arcs_between_nodes[n, m]])]

            if (subtours := find_subtours(selected_arcs)):
                for subtour in subtours:
                    # Convert subtours on the discretized vertex level \tilde{q}_i to subtours on a node level p_n
                    model.cbLazy(gp.quicksum(x[i,j] for n, m in product(subtour, subtour) for (i, j) in arcs_between_nodes[n, m]) <= len(subtour) - 1)

                return

            # Generate the visit path on the node level, that is 1 = n_1, n_2, ..., n_L = N 
            start_time_of_cut_generation = time.perf_counter()
            selected_arcs_dict = {n: m for n, m in selected_arcs}
            visit_order = [0]
            while visit_order[-1] != N - 1:
                visit_order.append(selected_arcs_dict[visit_order[-1]])
        
            L = len(visit_order)

            # NOTE: Check if the solution is feasible directly or if the CE path TSP found using SOCP is feasible wrt the original t_max value, 
            # otherwise perform another cut. Based on the Theorem from the paper, but strengthened whenever L >= 4.
            minimum_distance_through_nodes, _ = solve_CE_path_TSP_under_ordering(p[visit_order], r)
            if minimum_distance_through_nodes > t_max: # NOTE: per the preprocessing we must have L > 3.
                # NOTE: Add a cover cut if possible.
                key = frozenset(visit_order)
                if key not in seen:
                    seen.add(key)
                    dists[key], partial_order = solve_CE_path_TSP(p[visit_order], r, minimum_distance_through_nodes)
                    if dists[key] > t_max: 
                        strengthened_vertex_cover = [visit_order[i] for i in strengthen_minimum_infeasible_vertex_set(p[visit_order], r, t_max, partial_order)]
                        model.cbLazy(gp.quicksum(y[n] for n in strengthened_vertex_cover) <= len(strengthened_vertex_cover) - 1) 
                        cover_cut_times.append(time.perf_counter() - end_time_preprocessing)
                        return
                 
                    else:
                        incumbent_CEOP_solution_times.append(time.perf_counter() - end_time_preprocessing)
                        incumbent_CEOP_solution_scores.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))

            else:
                incumbent_CEOP_solution_times.append(time.perf_counter() - end_time_preprocessing)
                incumbent_CEOP_solution_scores.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))

            total_runtime_of_cut_generation += time.perf_counter() - start_time_of_cut_generation

    # Setup SOP discretization of the CEOP
    model = gp.Model("IPD discretization of CEOP")
    y = model.addVars(range(1, N - 1), vtype=GRB.BINARY, name="y") # NOTE: y[n] indicates that the node n is visited
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x") # NOTE: x[i, j] indicates that the arc (i, j) is utilized

    #model.setObjective(gp.quicksum(s[n] * y[n] for n in range(1, N - 1)) - 0.5 * min(s) / (t_max + (N - 2) * xi_IPD_scheme(r, K)) * gp.quicksum(d[i, j] * x[i, j] for (i, j) in arcs), GRB.MAXIMIZE)
    model.setObjective(gp.quicksum(s[n] * y[n] for n in range(1, N - 1)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(d[i, j] * x[i, j] for (i, j) in arcs) <= t_max + xi_IPD_scheme(r, K) * gp.quicksum(y[n] for n in range(1, N - 1)), name="adaptive adjusted maximum distance")
    model.addConstr(gp.quicksum(x[0, j] for j in outgoing[0]) == 1, name="departs from source")
    model.addConstr(gp.quicksum(x[i, M - 1] for i in incoming[M - 1]) == 1, name="arrives at sink")

    model.addConstrs((gp.quicksum(x[i, l] for i in incoming[l]) == gp.quicksum(x[l, j] for j in outgoing[l]) for l in range(1, M - 1)), name="flow conservation")
    
    model.addConstrs((gp.quicksum(x[i, j] for j in idx(n) for i in incoming[j]) == y[n] for n in range(1, N - 1)), name="in degree must equal visit indicator")
    
    # Preprocess and solve model.
    if time_limit is not None:
        model._start_time = time.perf_counter()
        model._time_limit = time_limit - (model._start_time - start_time)

    strengthen_model(model)
    model.params.LazyConstraints = 1

    model.update()
    root_model = model.relax()
    root_model.optimize()
    
    root_ub = root_model.ObjVal
    upper_bounds.append(root_ub)
    upper_bounds_times.append(time.perf_counter() - end_time_preprocessing)

    model.optimize(callback_with_SE)


    # Extract GOP solution, which may be converted to a CEOP solution
    visits = [0] + [n for n in range(1, N - 1) if y[n].x >= 0.5] + [N - 1]
    partial_ordering = solve_CE_path_TSP(p[visits], r, t_max)[-1]
    full_visit_order = [visits[i] for i in extract_order_from_partial_CE_path_TSP_order(p[visits], r, partial_ordering)]

    #selected_arcs = [(n, m) for n, m in product(range(N - 1), range(1, N)) if any([x[i, j].x >= 0.5 for (i, j) in arcs_between_nodes[n, m]])]

    #selected_arcs_dict = {n: m for n, m in selected_arcs}
    #visit_order = [0]
    #while visit_order[-1] != N - 1:
    #    visit_order.append(selected_arcs_dict[visit_order[-1]])

    #partial_ordering = solve_CE_path_TSP(p[visit_order], r, t_max)[-1]
    #full_visit_order = [visit_order[i] for i in extract_order_from_partial_CE_path_TSP_order(p[visit_order], r, partial_ordering)]
    
    return {"lb": model.objVal, 
            "gap": model.MIPGap, 
            "incumbent": solve_CE_path_TSP_under_ordering(p[full_visit_order], r)[-1],
            "total_runtime": time.perf_counter() - start_time, 
            "preprocessing_runtime": end_time_preprocessing - start_time,
            "cut_generation_runtime": total_runtime_of_cut_generation,
            "cover_cut_times": cover_cut_times,
            "upper_bounds": upper_bounds,
            "upper_bounds_times": upper_bounds_times,
            "incumbent_solution_times": incumbent_CEOP_solution_times,
            "incumbent_solution_scores": incumbent_CEOP_solution_scores}

if __name__ == "__main__":
    op_instance_id = "tsiligirides_problem_1_budget_20"
    path = os.path.join(PATH_TO_INSTANCES, f"{op_instance_id}.txt")
    p, s, t_max = load_instance(path)
    r = 0.5 
    K = 1 

    #import cProfile
    #cProfile.run("obj, gap, gamma, runtime = solve_BNC(p, s, t_max, r, K)")
    res = solve_BNC(p, s, t_max, r, K, time_limit=120) 
    gamma = res["incumbent"]
    
    #obj, gap, gamma, runtime = solve_BNC_old(p, s, t_max, r, K)
    #solve_BNC(p, s, t_max, r, K)
    #solve_BNC_old(p, s, t_max, r, K)
    print(f"Length of gamma is: {sum([np.linalg.norm(frm - to) for frm, to in zip(gamma[:-1], gamma[1:])]):.1f}, while T_max is: {t_max:.1f}")
    #print(f"Objective is {res["lb"]}")
    plot_solution(p, s, r, gamma, K)