from dataclasses import dataclass, field
from typing import Tuple, List, Set
from source.data_types import Matrix, Vector, Position
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
import numba as nb
from collections import defaultdict
from source.geometry import compute_distance_between_line_segment_and_point, compute_shortest_distances_to_vertices
from functools import lru_cache
import time

def solve_CE_path_TSP_under_ordering(p: Matrix, r: float) -> Tuple[float, Matrix]:
    """Solves the CE path TSP with the end points fixed, under a prespecified ordering, namely from p[0], p[1], ..., p[-1], terminating if a value less than t_max is found."""
    L = p.shape[0]
    if L == 2:
        return (np.linalg.norm(p[0] - p[-1]), p)

    model = gp.Model("CE path TSP under a prespecified ordering.")
    q = model.addVars(range(1, L - 1), 2, lb=-GRB.INFINITY, name="z") # NOTE: the actual waypoints
    t = model.addVars(L - 1, lb=0, name="t") # NOTE: represents the distances between consecutive points

    model.setObjective(gp.quicksum(t[l] for l in range(L - 1)), GRB.MINIMIZE)

    # NOTE: Constraints which ensures values of t_l >= actual distance between consequtive waypoints
    model.addQConstr((q[1,0] - p[0,0])*(q[1,0] - p[0,0]) + (q[1,1] - p[0,1])*(q[1,1] - p[0,1]) <= t[0]*t[0])
    for i in range(2, L - 1):
        model.addQConstr((q[i,0] - q[i-1,0])*(q[i,0] - q[i-1,0]) + (q[i,1] - q[i-1,1])*(q[i,1] - q[i-1,1]) <= t[i-1]*t[i-1])

    model.addQConstr((q[L-2,0] - p[-1,0])*(q[L-2,0] - p[-1,0]) + (q[L-2,1] - p[-1,1])*(q[L-2,1] - p[-1,1]) <= t[L-2]*t[L-2])

    # NOTE: Constraints which ensures: q_l \in \overline{B}(p_{n_l}, r)
    for i in range(1, L - 1):
        model.addQConstr((q[i, 0] - p[i, 0]) * (q[i, 0] - p[i, 0]) + (q[i, 1] - p[i, 1]) * (q[i, 1] - p[i, 1]) <= r**2)

    # FIXME: We may need to decrease the tolerances to ensure the correctness of the results.
    model.setParam("OutputFlag", 0)
    
    model.optimize()

    gamma = np.vstack([p[0], np.array([[q[i, 0].x, q[i, 1].x] for i in range(1, L - 1)]), p[-1]])
    #assert np.isclose(sum(np.linalg.norm(frm - to) for frm, to in zip(gamma[:-1], gamma[1:])), model.objVal)

    return model.ObjVal, gamma

@dataclass(order=True)
class Node:
    """Models a single node in the branch and bound search tree."""
    z: float
    order: List[int] = field(compare=False)
    qs: Matrix = field(compare=False)

def solve_CE_path_TSP(p: Matrix, r: float, upper_bound: float) -> Tuple[float, List[int]]:
    """Solves the close enough path TSP from p[0] to p[-1] going through all of the points in p. Using the branch and bound algorithm developed
       in 'A Branch-and-Bound Algorithm for the Close-Enough Traveling Salesman Problem' by Coutinho, Subramanian, Nascimento and Pessoa."""
    L = len(p)
    root_relaxation = Node(np.linalg.norm(p[0] - p[-1]), [0, L - 1], np.vstack([p[0], p[-1]]))
    open_nodes = [root_relaxation]
    heapq.heapify(open_nodes)

    upper_bound = np.inf 
    order_associated_with_upper_bound = None
    
    while len(open_nodes) != 0:
        node: Node = heapq.heappop(open_nodes)
        
        # Early stopping, since the nodes are sorted according to their objective values.
        if node.z > upper_bound:
            break 

        # In this case the node corresponds to a feasible solution to the CE path TSP.
        shortest_distances = compute_shortest_distances_to_vertices(node.order, node.qs, p, r)
        visits_mask = shortest_distances <= r
        if all(visits_mask) and node.z < upper_bound:
            upper_bound = node.z
            order_associated_with_upper_bound = node.order
            continue
        
        # Perform find the vertex to branch on and construct children, where the vertex is inserted in all of the different indices.
        vertex_to_branch_on = np.argmax(shortest_distances)
        for i in range(1, len(node.order)): 
            order = node.order[:]
            order.insert(i, vertex_to_branch_on)
            z, qs = solve_CE_path_TSP_under_ordering(p[order], r)
            if z < upper_bound:
                heapq.heappush(open_nodes, Node(z, order, qs))

    return upper_bound, order_associated_with_upper_bound

def strengthen_minimum_infeasible_vertex_set(p: Matrix, r: float, t_max: float, partial_order: List[int]) -> List[int]:
    """Strengthens the cover cut, but reducing the number of vertices in the minimum infeasible vertex set."""
    # Try to remove points from the cover until none can be removed without adhering to the maximum distance constraint t_max
    current_vertices = list(range(len(p)))

    while True:
        full_order_in_subproblem = extract_order_from_partial_CE_path_TSP_order(p[current_vertices], r, partial_order)
        original_full_order = [current_vertices[j] for j in full_order_in_subproblem]

        distances_with_point_i_plus_one_removed = [solve_CE_path_TSP_under_ordering(p[original_full_order[:i] + original_full_order[i + 1:]], r)[0] for i in range(1, len(original_full_order) - 1)]
        candidates = [i for i in range(1, len(original_full_order) - 1) if distances_with_point_i_plus_one_removed[i - 1] > t_max]
        if len(candidates) == 0:
            return current_vertices[1:-1]
        
        i = max(candidates, key=lambda i: distances_with_point_i_plus_one_removed[i - 1])
        dist, partial_order = solve_CE_path_TSP(p[original_full_order[:i] + original_full_order[i + 1:]], r, distances_with_point_i_plus_one_removed[i - 1])
        if dist <= t_max:
            return current_vertices[1:-1]

        else: # NOTE: We skip the index at i & remove it from the current vertices list.
            current_vertices.remove(original_full_order[i])

def extract_order_from_partial_CE_path_TSP_order(p: Matrix, r: float, order_associated_with_minimum_distance: List[int]) -> List[int]:
    """Extracts all of the necessary information from the CE-path-TSP result."""
    L = len(p)

    # Compute original order, form the CE-path-TSP result.
    qs = solve_CE_path_TSP_under_ordering(p[order_associated_with_minimum_distance], r)[1]
    insertion_indices_and_minimum_distances = {n: None for n in range(1, L - 1) if n not in order_associated_with_minimum_distance}
    for i in range(1, len(order_associated_with_minimum_distance)):
        for n in insertion_indices_and_minimum_distances.keys():
            d = compute_distance_between_line_segment_and_point(qs[i - 1], qs[i], p[n])
            if insertion_indices_and_minimum_distances[n] is None or d < insertion_indices_and_minimum_distances[n][1]:
                insertion_indices_and_minimum_distances[n] = (i, d)
        
    full_order = order_associated_with_minimum_distance
    vertex_insertion_map = defaultdict(list) 

    for n, (i, _) in insertion_indices_and_minimum_distances.items():
        vertex_insertion_map[i].append(n)
    
    for i in sorted(vertex_insertion_map.keys(), reverse=True):
        vertices = sorted(vertex_insertion_map[i], key=lambda n: -np.dot(qs[i] - qs[i - 1], qs[i] - p[n]) / np.dot(qs[i - 1] - qs[i], qs[i - 1] - qs[i]))
        full_order = full_order[:i] + vertices + full_order[i:]

    return full_order

if __name__ == "__main__":
    import os 
    from source.helpers import PATH_TO_INSTANCES, load_instance

    path = os.path.join(PATH_TO_INSTANCES, "tsiligirides_problem_1_budget_20.txt")
    original_p, s, t_max = load_instance(path)
    r = 0.5 

    order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    dist, partial_order = solve_CE_path_TSP(original_p[order], r, 1000)
    strengthened_vertex_cover = strengthen_minimum_infeasible_vertex_set(original_p[order], r, t_max, partial_order)
    print(f"res={original_p[[0] + strengthened_vertex_cover + [len(original_p) - 1]]}")

    p = np.vstack([[-4.8, -1.1], 2 * np.array([ [-2.1, 0.6], [-1.7, -0.5], [-1.4, 1.1], [-1, 0], [-0.4, 0.9], [0.4, -0.8], [1, 0], [1.2, 0.9], [1.9, -0.6], [2.4, 0.7]]), [7.0, 1.1]])
    print(solve_CE_path_TSP_under_ordering(p[[0, 2, 5, -3, -1]], r = 0.81))