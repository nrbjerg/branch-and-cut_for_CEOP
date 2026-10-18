import os 
import json
from source.helpers import load_instance, PATH_TO_BNC_RESULTS, PATH_TO_INSTANCES, RS, KS
from source.CE_path_TSP import solve_CE_path_TSP
from typing import Dict, Any
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

LAYOUTS = [f"tsiligirides_problem_{i}" for i in range(1, 4)]
results: Dict[str, Dict[str, Any]] = dict()
FILENAME_AND_EXTENSIONS = sorted(map(str, os.listdir(PATH_TO_BNC_RESULTS)))
for filename_and_extension in FILENAME_AND_EXTENSIONS:
    filename = os.path.splitext(filename_and_extension)[0]
    r, k = float(filename.split("_")[-2]), int(filename.split("_")[-1])
    with open(os.path.join(PATH_TO_BNC_RESULTS, filename_and_extension), "r") as file:
        results[filename] = json.load(file)

def plot_incumbents_and_cover_cuts(instance_id: str):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    axs = axs.ravel()
    plt.style.use("seaborn-v0_8-ticks")
    for i, k in enumerate(KS):
        ax1 = axs[i]
        with open(os.path.join(PATH_TO_BNC_RESULTS, f"{instance_id}_{k}.json"), "r") as file:
            res = json.load(file)
            solve_runtime = res["total_runtime"] - res["preprocessing_runtime"]

            ax1.plot(res["incumbent_solution_times"], res["incumbent_solution_scores"], marker="o", label="CEOP Incumbent")
            # We dont want a mark at the end.
            ax1.plot([res["incumbent_solution_times"][-1], solve_runtime], [res["incumbent_solution_scores"][-1], res["incumbent_solution_scores"][-1]], color="tab:blue")

            ax1.plot(res["upper_bounds_times"], res["upper_bounds"], marker="s", label="MILP UB", color="tab:orange", zorder=2)
            if res["gap"] <= 0.0:
                ax1.plot([res["upper_bounds_times"][-1], solve_runtime], [res["upper_bounds"][-1], res["incumbent_solution_scores"][-1]], color="tab:orange")
            else:
                ax1.plot([res["upper_bounds_times"][-1], solve_runtime], [res["upper_bounds"][-1], res["upper_bounds"][-1]], color="tab:orange")
            # We dont want a mark at the end.
            ax1.set_ylabel("Score")
            ax1.set_xlabel("Time (s)")

            ax2 = ax1.twinx()
            ax2.step([0] + res["cover_cut_times"] + [solve_runtime], list(range(len(res["cover_cut_times"]) + 1)) + [len(res["cover_cut_times"]) + 1], where="post", label="Cover Cuts", color="tab:green")
            ax2.set_ylabel("Number of Cover Cuts")
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

            _, _, layout, _, t_max, r = instance_id.split("_")
            ax1.set_title(f"Trace of Set {layout} with $t_{{\max}} = {float(t_max):g}, r = {float(r):g}$ and $K = {int(k)}$")
    
            ax1.set_xlim(0, solve_runtime)

    handels1, labels1 = ax1.get_legend_handles_labels()
    handels2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(handels1 + handels2, labels1 + labels2, loc="lower center", ncol=3)

    fig.tight_layout()
    plt.show()

def plot_runtimes():
    # Generate plots
    plt.rcParams['figure.figsize'] = [10, 7]
    plt.style.use("seaborn-v0_8-ticks")
    fig, axs = plt.subplots(3, 4)
    for idx, layout in enumerate(LAYOUTS):
        for jdx, r in enumerate(RS):
            for l, k in enumerate(KS):
                keys = [key for key in results.keys() if key.startswith(layout) and int(key.split("_")[-1]) == k and float(key.split("_")[-2]) == r]
                #data = [len(results[key]["cover_cut_times"]) for key in sorted(keys)] 
                data = [results[key]["total_runtime"] for key in sorted(keys)]
                axs[idx, jdx].plot([int(key.split("_")[-3]) for key in sorted(keys)], data, label=f"$K = {k}$")

            axs[idx, jdx].set_ylabel("Runtime (s)")
            axs[idx, jdx].set_xlabel("$T_{\\max}$")
            axs[idx, jdx].set_title(f"Set {layout[-1]} ($r = {r}$)")

    handels, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handels, labels, loc="lower center", ncol=5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
            
def generate_tables():
    # Construct tables 
    for layout in [f"tsiligirides_problem_{i}" for i in range(1, 4)]:
        print(f"Results for {layout}")
        results_for_layout = {k: v for k, v in results.items() if k.startswith(layout)}
        
        for r in RS:
            print(f"\n{r=}")
            print("$T_{\max}$ & $K$ & LB & Gap & Total & Pre & Master & Sub & $\# CC$ \\\\")
            print("\\midrule")
            keys_for_r = sorted([key for key in results_for_layout.keys() if float(key.split("_")[-2]) == r])
            for i in range(len(keys_for_r) // 4):
                group = keys_for_r[4 * i:4*(i + 1)]
                best_string, best_pair = None, (np.inf, np.inf)
                for key in group:
                    t_max, k = int(key.split("_")[-3]), int(key.split("_")[-1][0])

                    res = results_for_layout[key]
                    parameter_string = f"{t_max} & {k}"
                    objective_string = f"{res["lb"]:g} & " + (f"{res["gap"] * 100:.1f}\\%" if res["gap"] != 0.0 else "-")
                    master_runtime = res["total_runtime"] - res["preprocessing_runtime"] - res["cut_generation_runtime"]
                    runtime_string = f"{res["total_runtime"]:.1f} & {res["preprocessing_runtime"]:.1f} & {master_runtime:.1f} & {res["cut_generation_runtime"]:.1f}"
                    misc_information = f"{len(res["cover_cut_times"])} & {len(res["incumbent_solution_scores"])}"

                    
                    # Prefer converged solutions; among those, minimize runtime or if no K has converged, minimize gap.
                    if (res["gap"] == 0.0 and (best_pair[0] != 0.0 or res["total_runtime"] < best_pair[1])) or (best_pair[0] != 0.0 and (res["gap"] < best_pair[0] or (res["gap"] == best_pair[0] and res["total_runtime"] < best_pair[1]))):
                        best_string = " & ".join([parameter_string, objective_string, runtime_string, misc_information]) + " \\\\"
                        best_pair = (0.0, res["total_runtime"])

                print(best_string)
        
        print("\n\n")

    plt.clf()
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['figure.dpi'] = 300
    plt.style.use("seaborn-v0_8-ticks")
    for k in [1, 3, 5, 7]:
        results_for_discretization_level = {key: val for key, val in results.items() if key.endswith(str(k))}
        
        incumbent_solutions, cover_cuts, all_runtimes, gaps = [], [], [], []
        gaps_over_time = []
        for res in results_for_discretization_level.values():
            cover_cuts.append(len(res["cover_cut_times"]))
            incumbent_solutions.append(len(res["incumbent_solution_scores"]))
            if res["gap"] == 0.0:
                all_runtimes.append([res["total_runtime"], res["preprocessing_runtime"], res["total_runtime"] - res["preprocessing_runtime"] - res["cut_generation_runtime"], res["cut_generation_runtime"]])
            else:
                gaps.append(res["gap"])
            
            def gap(t):
                if any([t_prime <= t for t_prime in res["upper_bounds_times"]]):
                    ub_at_t = max([ub for ub, t_prime in zip(res["upper_bounds"], res["upper_bounds_times"]) if t_prime <= t])
                    lb_at_t = max([lb for lb, t_prime in zip(res["incumbent_solution_scores"], res["incumbent_solution_times"]) if t_prime <= t])
                    return (ub_at_t - lb_at_t) / ub_at_t

                else:
                    return (res["upper_bounds"][0] - res["incumbent_solution_scores"][0]) / res["upper_bounds"][0]

            gaps_over_time.append([gap(t) for t in range(1, 14000)])

        plt.plot(range(1, 14000), np.mean(gaps_over_time, axis=0) * 100, label=f"$K = {k}$")
        all_runtimes = np.array(all_runtimes)
        # Preprocessing runtimes: 
        # print(f"{np.median(all_runtimes[:, 1]):.1f}, {np.mean(all_runtimes[:, 1]):.1f}, {np.std(all_runtimes[:, 1]):.1f}")

        # Runtime and GAP information: 
        #runtimes = all_runtimes[:, 0]
        #gaps_in_percent = 100 * np.array(gaps)
        #runtime_string = f"{np.median(runtimes):.1f} & {np.mean(runtimes):.1f} & {np.std(runtimes):.1f}"
        #gap_string = f"{np.min(gaps_in_percent):.1f}\\% & {np.median(gaps_in_percent):.1f}\\% & {np.mean(gaps_in_percent):.1f}\\% & {np.std(gaps_in_percent):.1f}\\%"
        # print(f"{k} & {len(runtimes)} & {runtime_string} & {gap_string} \\\\")

        cover_cut_string = f"{np.max(cover_cuts)} & {np.median(cover_cuts):.1f} & {np.mean(cover_cuts):.1f} & {np.std(cover_cuts):.1f}"
        incumbent_solutions_string = f"{np.max(incumbent_solutions)} & {np.median(incumbent_solutions):.1f} & {np.mean(incumbent_solutions):.1f} & {np.std(incumbent_solutions):.1f}"
        print(f"{k} & {cover_cut_string} & {incumbent_solutions_string}")

    plt.legend()
    plt.ylabel("Mean Gap (%)", fontsize="large")
    plt.xlabel("Execution Time (s) in Log-Scale", fontsize="large")
    plt.xlim(1, 13999)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("mean_gap_against_execution_time.png")

def plot_influence_of_radius_on_runtimes():
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['figure.dpi'] = 300
    plt.style.use("seaborn-v0_8-ticks")
    fig, ax = plt.subplots()

    positions = np.arange(1, 4)                 # Layout 1, 2, 3
    offsets = [-0.30, -0.10, 0.10, 0.30]       # r = 0.5, 1, 1.5, 2
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    legend_handles = []

    for r, dx, color in zip(RS, offsets, colors):

        data = []

        for layout in [f"tsiligirides_problem_{i}" for i in range(1, 4)]:
            results_for_layout = {
                k: v for k, v in results.items() if k.startswith(layout)
            }

            keys_for_r = sorted(
                key for key in results_for_layout
                if float(key.split("_")[-2]) == r
            )

            data.append([
                results_for_layout[key]["total_runtime"]
                for key in keys_for_r
            ])

        bp = ax.boxplot(
            data,
            positions=positions + dx,
            widths=0.16,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False
        )

        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor="black")
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.2)
        for whisker in bp["whiskers"]:
            whisker.set(color="black")
        for cap in bp["caps"]:
            cap.set(color="black")

        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=color, edgecolor="black")
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Set 1", "Set 2", "Set 3"])
    ax.set_ylabel("Runtime (s) in Log-Scale", fontsize="large")
    ax.set_xlabel("Layout", fontsize="large")

    ax.legend(
        legend_handles,
        [r"$r=0.5$", r"$r=1$", r"$r=1.5$", r"$r=2$"],
    )
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("runtimes_against_radius.png")

def generate_can_visit_all_table():
    FILENAME_AND_EXTENSIONS = sorted(map(str, os.listdir(PATH_TO_INSTANCES)))
    lengths = {}
    for r in RS:
        count = np.zeros(3, dtype=np.int64)
        total_count = np.zeros(3, dtype=np.int64)
        for filename_and_extension in FILENAME_AND_EXTENSIONS:
            print(filename_and_extension)
            layout = int(filename_and_extension.split("_")[2]) - 1
            total_count[layout] += 1
            key = f"{layout}, {r}"
            p, s, t_max = load_instance(os.path.join(PATH_TO_INSTANCES, filename_and_extension))
            if key not in lengths.keys():
                lengths[key] = solve_CE_path_TSP(p, r, np.inf)[0]
            
            if lengths[key] <= t_max:
                count[layout] += 1

        print(f"${r=}$ & {count[0]}/{total_count[0]} & {count[1]}/{total_count[1]} & {count[2]}/{total_count[2]} \\\\")

    print(lengths)

import numpy as np
import matplotlib.pyplot as plt

def compute_runtime_percentages():
    total_runtimes = []
    milp_runtimes = [] 
    sub_problem_runtimes = []
    preprocessing_runtimes = []
    for res in results.values():
        total_runtimes.append(res["total_runtime"])
        milp_runtime = res["total_runtime"] - res["preprocessing_runtime"] - res["cut_generation_runtime"] 
        milp_runtimes.append(milp_runtime)
        sub_problem_runtimes.append(res["cut_generation_runtime"])
        preprocessing_runtimes.append(res["preprocessing_runtime"])

    print(f"Mean runtime percentages: {np.mean(milp_runtimes):.1f}, {np.mean(sub_problem_runtimes):.1f}, {np.mean(preprocessing_runtimes):.1f}")
    print(f"Median runtime percentages: {np.median(milp_runtimes):.1f}, {np.median(sub_problem_runtimes):.1f}, {np.median(preprocessing_runtimes):.1f}")

if __name__ == "__main__":
    plot_runtimes()
    compute_runtime_percentages()
    plot_influence_of_radius_on_runtimes()
    generate_tables()
    instance_id = "tsiligirides_problem_3_budget_075_0.5"  
    plot_incumbents_and_cover_cuts(instance_id)
    generate_can_visit_all_table()