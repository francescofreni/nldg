"""
DISCLAIMER:
This script is intended for systems with more than 50 CPU cores.
Running it on machines with fewer cores may lead to memory issues.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from nldg.utils import gen_data_v6
from adaXT.random_forest import RandomForest


def plot_runtime(
    cores: list[int],
    cp_times: list[float],
    extragradient_times: list[float],
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # CP Plot
    axs[0].plot(
        cores,
        cp_times,
        color="lightskyblue",
        marker="o",
        markeredgecolor="white",
    )
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title("CP Runtime vs Number of Cores")
    axs[0].grid(True, linewidth=0.2)

    # Extragradient Plot
    axs[1].plot(
        cores,
        extragradient_times,
        color="lightskyblue",
        marker="o",
        markeredgecolor="white",
    )
    axs[1].set_xlabel("Number of Cores")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].set_title("Extragradient Runtime vs Number of Cores")
    axs[1].grid(True, linewidth=0.2)

    plt.tight_layout()

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(parent_dir, "results")
    plots_dir = os.path.join(results_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    outpath = os.path.join(plots_dir, "runtime_vs_cores.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    dtr = gen_data_v6(n=1000, noise_std=0.5)
    Xtr = np.array(dtr.drop(columns=["E", "Y"]))
    Ytr = np.array(dtr["Y"])
    Etr = np.array(dtr["E"])

    n_estimators = 50
    min_samples_leaf = 30
    random_state = 42

    cores = [1, 2, 4, 8, 16, 32, 50]
    cp_times = []
    extragradient_times = []

    for n_jobs in cores:
        print(f"\nRunning CP with n_jobs={n_jobs}")
        rf = RandomForest(
            "Regression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=random_state,
        )
        rf.fit(Xtr, Ytr)

        start = time.time()
        rf.modify_predictions_trees(Etr, n_jobs=n_jobs)
        end = time.time()
        cp_times.append(end - start)
        print(f"  Time: {end - start:.2f}s")

        print(f"Running Extragradient with n_jobs={n_jobs}")
        rf = RandomForest(
            "Regression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=random_state,
        )
        rf.fit(Xtr, Ytr)

        start = time.time()
        rf.modify_predictions_trees(
            Etr, opt_method="extragradient", n_jobs=n_jobs
        )
        end = time.time()
        extragradient_times.append(end - start)
        print(f"  Time: {end - start:.2f}s")

    # Plotting
    plot_runtime(
        cores=cores,
        cp_times=cp_times,
        extragradient_times=extragradient_times,
    )
