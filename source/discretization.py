from source.data_types import Position, Matrix
import numpy as np

def compute_IPD_points(p_n: Position, r: float, K: int) -> Matrix:
    """Returns the interior point discretization points for a given position p and radius r"""
    if K == 1:
        return np.expand_dims(p_n, axis=0)

    delta_theta = 2 * np.pi / K 
    thetas = np.array([delta_theta * k for k in range(K)])
    PD_points = np.expand_dims(p_n, axis=1) + r * np.stack([np.cos(thetas), np.sin(thetas)])

    IPD_points = np.hstack((0.5 * (PD_points[:, :-1] + PD_points[:, 1:]), np.expand_dims(0.5 * (PD_points[:, 0] + PD_points[:, -1]), axis=0).T))
    return IPD_points.T

def xi_IPD_scheme(r: float, K: int) -> float:
    """Returns the discretization error associated with the IPD scheme."""
    if K == 1:
        return 2 * r
    else:
        return 2 * r * np.sin(np.pi/K)


