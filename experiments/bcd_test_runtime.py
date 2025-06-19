import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from nldg.utils import max_mse
from adaXT.random_forest import RandomForest

N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42


def assign_quadrant(
    Z: pd.DataFrame,
) -> np.ndarray:
    lat, lon = Z["Latitude"], Z["Longitude"]
    north = lat >= 35
    south = ~north
    east = lon >= -120
    west = ~east

    env = np.zeros(len(Z), dtype=int)
    env[south & west] = 0  # SW
    env[south & east] = 1  # SE
    env[north & west] = 2  # NW
    env[north & east] = 3  # NE
    return env


def plot_bcd_runtime(block_sizes, bcd_times, baseline_time, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(
        block_sizes,
        bcd_times,
        marker="o",
        color="lightskyblue",
        markeredgecolor="white",
        label="BCD",
    )
    plt.axhline(
        y=baseline_time, color="orange", linestyle="--", label="Non-BCD"
    )
    plt.xlabel("Block Size")
    plt.ylabel("Runtime (s)")
    plt.grid(True, linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_block_size.png"), dpi=300)
    plt.close()


def plot_mse_by_method(
    mse_rf, mse_non_bcd, mse_bcd_dict, block_sizes, out_dir
):
    env_labels = ["Env 1", "Env 2", "Env 3"]
    methods = ["RF", "Non-BCD"] + [f"BCD (bs={b})" for b in block_sizes]
    num_methods = len(methods)
    env_colors = ["lightskyblue", "orange", "mediumpurple"]

    # Stack MSEs per method (rows) × envs (columns)
    mse_matrix = [mse_rf, mse_non_bcd, *[mse_bcd_dict[b] for b in block_sizes]]

    x = np.arange(num_methods)
    width = 0.2

    plt.figure(figsize=(15, 6))
    for env_idx in range(3):
        mse_vals = [m[env_idx] for m in mse_matrix]
        plt.bar(
            x
            + width * (env_idx - 1),  # center the bars around the method index
            mse_vals,
            width,
            label=env_labels[env_idx],
            color=env_colors[env_idx],
        )

    plt.xticks(x, methods)
    plt.ylabel("MSE")
    plt.xlabel("Method")
    plt.grid(True, axis="y", linewidth=0.3)
    plt.legend()
    plt.ylim(0.3, 0.55)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bcd_mse_per_environment.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    Z = X[["Latitude", "Longitude"]]
    X = X.drop(["Latitude", "Longitude"], axis=1)

    env = assign_quadrant(Z)

    print("Fitting standard Random Forest...", end=" ", flush=True)
    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf.fit(X, y)
    print("Done!", flush=True)
    fitted_rf = rf.predict(X)
    mse_envs_rf, _ = max_mse(y, fitted_rf, env, ret_ind=True)

    # Non-BCD Minimax
    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf.fit(X, y)
    print("Modifying predictions with non-BCD...", end=" ", flush=True)
    start = time.time()
    rf.modify_predictions_trees(env)
    end = time.time()
    print("Done!", flush=True)
    non_bcd_time = end - start
    fitted_minimax = rf.predict(X)
    mse_envs_minimax, _ = max_mse(y, fitted_minimax, env, ret_ind=True)

    # BCD Variants
    block_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    bcd_times = []
    mse_envs_bcd = {}

    for b in block_sizes:
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X, y)
        print(
            f"Modifying predictions with BCD (bs {b})...", end=" ", flush=True
        )
        start = time.time()
        rf.modify_predictions_trees(env, bcd=True, patience=1, block_size=b)
        end = time.time()
        print("Done!", flush=True)
        bcd_times.append(end - start)

        fitted_bcd = rf.predict(X)
        mse_envs, _ = max_mse(y, fitted_bcd, env, ret_ind=True)
        mse_envs_bcd[b] = mse_envs

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(parent_dir, "results")
    plots_dir = os.path.join(results_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)

    plot_bcd_runtime(block_sizes, bcd_times, non_bcd_time, plots_dir)
    plot_mse_by_method(
        mse_envs_rf, mse_envs_minimax, mse_envs_bcd, block_sizes, plots_dir
    )
