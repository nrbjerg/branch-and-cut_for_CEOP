import numba as nb
from source.data_types import Position, Matrix, Vector
import numpy as np
from typing import List

@nb.njit(inline="always")
def compute_distance_between_line_segment_and_point(q0: Position, q1: Position, c: Position) -> float:
    """Computes the shortest distance between a line-segment between q0 and q1 and a point c.""" 
    theta = -np.dot(q1 - q0, q0 - c) / np.dot(q0 - q1, q0 - q1)
    if theta <= 0:
        return np.linalg.norm(q0 - c)
    elif theta >= 1:
        return np.linalg.norm(q1 - c)
    else:
        return np.linalg.norm((1 - theta) * q0 + theta * q1 - c)


@nb.njit(parallel=False)
def compute_shortest_distances_to_vertices(order: List[int], qs: Matrix, p: Matrix, r: float) -> Vector:
    """Checks if the path going from p[0], through the qs, ending at p[-1], visits the disks."""
    n = len(p)

    # Boolean lookup instead of `i in order`
    visited = np.zeros(n, np.uint8)
    for idx in order:
        visited[idx] = 1

    distances = np.empty(n)

    for i in nb.prange(n):
        if visited[i]:
            distances[i] = 0.0
            continue

        if i == 0 or i == n - 1:
            distances[i] = 0.0
            continue

        best = np.inf
        for j in range(len(qs) - 1):
            d = compute_distance_between_line_segment_and_point(qs[j], qs[j+1], p[i])
            if d < best:
                best = d

            if d < r:
                break

        distances[i] = best

    return distances

@nb.njit(inline="always")
def compute_closest_point_in_uncovered_disk_to_line_segment(q_prev: Position, q_next: Position, p: Position, r: float) -> Position:
    """Computes the closest point in B(p, r) to the line segment between q_prev and q_next."""
    v = q_next - q_prev
    norm_of_v_sq = np.dot(v, v)

    if norm_of_v_sq == 0: # NOTE: Degenerate segment
        proj_of_p_n_onto_seg = q_next
    else: 
        # Projection of p_star onto the infinite line
        tau = min(1.0, max(np.dot(p - q_prev, v) / norm_of_v_sq, 0.0))
        proj_of_p_n_onto_seg = q_prev + tau * v

    proj_of_p_n_onto_seg_to_p_k_star = proj_of_p_n_onto_seg - p
    dist = np.linalg.norm(proj_of_p_n_onto_seg_to_p_k_star)
                
    # NOTE: Here we utilize that the disc is uncovered.
    return p + r * proj_of_p_n_onto_seg_to_p_k_star / dist