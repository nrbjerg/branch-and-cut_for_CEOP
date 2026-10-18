from typing import Tuple
import numpy as np
from source.data_types import Matrix, Vector
from source.CE_path_TSP import solve_CE_path_TSP_under_ordering
from source.discretization import xi_IPD_scheme, compute_IPD_points
from itertools import product
from scipy.spatial import distance_matrix

def preprocess_CEOP(p: Matrix, s: Vector, t_max: float, r: float) -> Tuple[Matrix, Vector, float, float]:
    """Preprocess the CEOP, by removing points whose disk cannot be reached on a path from the source to the sink"""
    preprocessed_p, preprocessed_s = [], []
    for n in reversed(range(len(s))):
        if solve_CE_path_TSP_under_ordering(p[[0, n, -1]], r)[0] <= t_max:
            preprocessed_p.append(p[n])
            preprocessed_s.append(s[n])
        
    return np.array(preprocessed_p), np.array(preprocessed_s), t_max, r

def preprocess_SOP(p: Matrix, s: Vector, t_max: float, r: float, K: int) -> Tuple[Matrix, Matrix, Matrix]:
    """Implements all of the preprocessing described in the paper. Returns a preprocessed adjacency matrix for the discretized SOP"""
    # Index stuff
    N = p.shape[0]
    M = 2 + (N - 2) * K
    idx = lambda n: ([0] if n == 0 else [M - 1]) if n == 0 or n == N - 1 else range(1 + (n - 1) * K, 1 + n * K) # NOTE: Corresponds to \mathcal{I} from the paper.

    q = np.vstack((p[0], *(compute_IPD_points(p[n], r, K) for n in range(1, N - 1)), p[-1]))
    d = distance_matrix(q, q)

    adj = np.ones((M, M), dtype=np.bool)

    # Cannot go back to the source and cannot leave the sink.
    adj[:, 0] = False 
    adj[-1, :] = False

    ## Adjacency matrix preprocessing based on CE-path-TSP
    for n, m in product(range(1, N - 1), range(1, N - 1)):
        if n == m or solve_CE_path_TSP_under_ordering(p[[0, n, m, -1]], r)[0] > t_max:
            for (i, j) in product(idx(n), idx(m)):
                adj[i, j] = False

    ## Check if there is always a shorter path through the disks
    changed = True
    while changed:
        changed = False
        for n_prime, n in product(range(N - 1), range(1, N - 1)):
            # Find the shortest paths through the discretization points of n, from n' to n*, if any arc (i, j) is not utilized by any such path then we may remove it
            vertex_of_shortest_intermediate_path = dict()
            for n_star in range(1, N):
                if len(set([n_prime, n, n_star])) != 3:
                    continue
                
                for (i, l) in product(idx(n_prime), idx(n_star)):
                    d_min = np.inf
                    for j in idx(n):
                        if adj[i, j] and adj[j, l] and d[i, j] + d[j, l] < d_min:
                            d_min = d[i, j] + d[j, l]
                            vertex_of_shortest_intermediate_path[(i, l)] = j

            utilized_arc_from_n_prime_to_n = set()
            for (i, _), j in vertex_of_shortest_intermediate_path.items():
                utilized_arc_from_n_prime_to_n.add((i, j))
                
            for (i, j) in product(idx(n_prime), idx(n)):
                if adj[i, j] and (i,j) not in utilized_arc_from_n_prime_to_n:
                    changed = True
                    adj[i, j] = False

        # NOTE: if no arcs are going into i then we do not need to have any arcs going from i to the sink
        for i in range(1, M - 1): 
            if adj[i, -1] and any(adj[:, i]) is False:
                adj[i, -1] = False

    print(f"Preprocessed version contains: {(np.sum(adj) / (M * M) * 100):.1f}% of the arcs of the original.")
    return q, d, adj