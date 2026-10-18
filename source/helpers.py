from typing import Optional, Set, List, Tuple
from source.data_types import Matrix, Vector
from source.discretization import compute_IPD_points
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import numpy as np
import os 

TIME_LIMIT = 3600 * 4
RS = [0.5, 1.0, 1.5, 2.0]
KS = [1, 3, 5, 7]

PATH_TO_BNC_RESULTS = os.path.join(os.getcwd(), "archive", "CEOP_BnC")
PATH_TO_MIQCP_RESULTS = os.path.join(os.getcwd(), "archive", "CEOP_MICQP")
PATH_TO_INSTANCES = os.path.join(os.getcwd(), "resources", "OP")

# ------------------------------------------------------------ Helper functions ------------------------------------------------------------ #
def plot_solution(p: Matrix, s: Vector, r: float, gamma: Optional[Matrix] = None, K: Optional[int] = None, title: str = None):
    """plots the solution gamma, within the context of the CEOP."""
    plt.gca().set_aspect("equal")

    plt.scatter(p[1:-1, 0], p[1:-1, 1], s=1 + s[1:-1], c="black")
    for center in p[1:-1]:
        plt.gca().add_patch(mpl.patches.Circle(center, radius=r, fill=False, linestyle="dotted"))

        if K != None: # Plot the IPD points
            for q in compute_IPD_points(center, r, K):
                plt.scatter(q[0], q[1], c="red")

    plt.scatter([p[0, 0]], [p[0, 1]], s = 40, c="black", marker="s", zorder=2)
    plt.scatter([p[-1, 0]], [p[-1, 1]], s = 40, c="black", marker="D", zorder=2)

    if gamma is not None:
        plt.plot(gamma[:, 0], gamma[:, 1], zorder=1, color="black")#, linestyle="dotted", marker="x", zorder=1)
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    #plt.savefig("CEOP_example")
    plt.show()
    
def find_subtours(selected_arcs: List[Tuple[int, int]]) -> List[List[int]]:
    """Finds all of the subtours within the graph imposed by the selected arcs (on the node level)."""
    succ = {i: j for i, j in selected_arcs}
    seen = set()
    tours = []

    for current in succ.keys():
        if current in seen:
            continue

        path = []
        while current not in path and current in succ:
            path.append(current)
            current = succ[current]

        # In this case we have found a loop, so we find the first occurrence of current and take the path from there to the next occurrence of current.
        if current in path:
            tours.append(path[path.index(current):])

        seen.update(path)

    return tours

def load_instance(path: str) -> Tuple[Matrix, Vector, float]:
    """Loads the OP instance from the file at the path provided"""
    with open(path, "r") as file:
        lines = file.readlines()

    t_max = float(lines[0].split()[0])
    nodes = np.array([list(map(float, line.split())) for idx, line in enumerate(lines[1:], start=1) if idx != 2] + [list(map(float, lines[2].split()))])
    p = nodes[:, :2]
    s = nodes[:, 2]

    return p, s, t_max