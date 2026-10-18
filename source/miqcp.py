from source.data_types import Matrix, Vector
from typing import Tuple, Optional
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import numpy as np
from source.helpers import find_subtours
from source.preprocessing import preprocess_CEOP

def solve_MIQCP(p: Matrix, s: Vector, t_max: float, r: float, time_limit: Optional[float] = None) -> Tuple[float, Matrix, float]:
    """Solves the CEOP directly using Gurobi to solve the associated Mixed Integer Quadratic Constrained Program (MIQCP)"""
    p, s, t_max, r = preprocess_CEOP(p, s, t_max, r)

    # Index stuff
    N = p.shape[0]
    def callback(model: gp.Model, where):
        """Checks if the solution found is feasible, in which case we have found a feasible solution for the CEOP, if we can check if the CE path TSP
           solution is feasible within the CEOP framework i.e. does it obey the imposed maximum distance budget t_max? If not we may include a sub-tour 
           elimination style cut, we may also include such a cut if the solution is not connected etc."""
        if where == GRB.Callback.MIPSOL:
            # Check if the solution is connected, otherwise add a cut for each sub-tour. 
            x_vals = model.cbGetSolution(x)
            selected_arcs = [(n, m) for n, m in product(range(N - 1), range(1, N)) if x_vals[n, m] >= 0.5]

            if (subtours := find_subtours(selected_arcs)):
                for subtour in subtours:
                    # Convert subtours on the discretized vertex level \tilde{q}_i to subtours on a node level p_n
                    model.cbLazy(gp.quicksum(x[n,m] for n, m in product(subtour, subtour)) <= len(subtour) - 1)

                return

    # Setup GOP discretization of the CEOP
    model = gp.Model("CEOP (MIQCP version)")
    y = model.addVars(range(1, N - 1), vtype=GRB.BINARY, name="y") # NOTE: y[n] indicates that the node n is visited
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x") # NOTE: x[n, m] indicates that the arc (n, m) is utilized
    q = model.addVars(range(1, N - 1), 2, lb = -GRB.INFINITY, name="q") # NOTE: q[n] is the waypoint within \overline{B}(p[n], r)
    t = model.addVars(N, N, name="t") # NOTE: t[n,m] is the distance between q[n] and q[m].

    model.setObjective(gp.quicksum(s[n] * y[n] for n in range(1, N - 1)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, N)) == 1, name="departs from source")
    model.addConstr(gp.quicksum(x[i, N-1] for i in range(N - 1)) == 1, name="arrives at sink")
    model.addConstrs((gp.quicksum(x[i, l] for i in range(N - 1) if i != l) == gp.quicksum(x[l, j] for j in range(1, N) if j != l) for l in range(1, N - 1)), name="flow conservation")
    model.addConstrs((y[j] == gp.quicksum(x[i, j] for i in range(N - 1) if i != j) for j in range(1, N - 1)), name="visit constraint")

    adj = np.ones((N, N), dtype=np.bool)
    for n, m in product(range(1, N - 1), range(1, N - 1)):
        if solve_CE_path_TSP_under_ordering(p[[0, n, m, -1]], r)[0] > t_max:
            adj[n, m] = 0

    # Remove variables x based on adjacency matrix.
    model.addConstr(gp.quicksum(x[n, m] for n, m in product(range(N), range(N)) if adj[n, m] == 0) == 0, name="Infeasible arcs")

    # Add exclusionary constraints based on adjacency matrix.
    for n in range(1, N - 1):
        exclusionary_nodes = [m for m in range(1, N - 1) if max(adj[m, n], adj[n, m]) == 0]

        if exclusionary_nodes:
            model.addConstr(y[n] - 1 <= gp.quicksum(y[m] for m in exclusionary_nodes), name=f"Exclusionary nodes")

    model.addQConstr(gp.quicksum(t[i, j] * x[i, j] for i, j in product(range(N - 1), range(1, N))) <= t_max, name="maximum distance")
    for i, j in product(range(N - 1), range(1, N)):
        if i == j or adj[i, j] == 0:
            continue

        if i != 0 and j != N - 1:
            model.addQConstr((q[i,0] - q[j,0])*(q[i,0] - q[j,0]) + (q[i,1] - q[j,1])*(q[i,1] - q[j,1]) <= t[i, j]*t[i, j])
        elif i == 0 and j == N - 1:
            model.addQConstr((p[i,0] - p[j,0])*(p[i,0] - p[j,0]) + (p[i,1] - p[j,1])*(p[i,1] - p[j,1]) <= t[i, j]*t[i, j])
        elif i == 0:
            model.addQConstr((p[i,0] - q[j,0])*(p[i,0] - q[j,0]) + (p[i,1] - q[j,1])*(p[i,1] - q[j,1]) <= t[i, j]*t[i, j])
        else:
            model.addQConstr((q[i,0] - p[j,0])*(q[i,0] - p[j,0]) + (q[i,1] - p[j,1])*(q[i,1] - p[j,1]) <= t[i, j]*t[i, j])
    
    for i in range(1, N - 1):
        model.addQConstr((q[i, 0] - p[i, 0]) * (q[i, 0] - p[i, 0]) + (q[i, 1] - p[i, 1]) * (q[i, 1] - p[i, 1]) <= r**2)

    #model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    model.params.LazyConstraints = 1
    model.optimize(callback)

    selected_arcs = {i: j for i, j in product(range(N), range(N)) if x[i, j].x >= 0.5}
    route = [0]
    while route[-1] != N - 1:
        route.append(selected_arcs[route[-1]])

    gamma = np.array([p[0]] + [[q[n, 0].x, q[n, 1].x] for n in route[1:-1]] + [p[-1]])
    
    return model.ObjVal, model.MIPGap, gamma, model.Runtime

