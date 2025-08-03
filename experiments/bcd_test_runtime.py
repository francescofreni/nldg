import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from nldg.utils import max_mse
from adaXT.random_forest import RandomForest
from sklearn.model_selection import train_test_split

N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
NAME_RF = "MaxRM-RF"
B = 20
VAL_PERCENTAGE = 0.3
BLOCK_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def assign_quadrant(
    Z: pd.DataFrame,
) -> np.ndarray:
    lat, lon = Z["Latitude"], Z["Longitude"]

    # north = lat >= 35
    # south = ~north
    # east = lon >= -120
    # west = ~east
    #
    # env = np.zeros(len(Z), dtype=int)
    # env[south & west] = 0  # SW
    # env[south & east] = 1  # SE
    # env[north & west] = 2  # NW
    # env[north & east] = 3  # NE

    west = lon < -121.5
    east = ~west
    sw = (lat < 38) & west
    nw = (lat >= 38) & west
    lat_thr = 34.5
    se = (lat < lat_thr) & east
    ne = (lat >= lat_thr) & east

    env = np.zeros(len(Z), dtype=int)
    env[sw] = 0  # SW
    env[se] = 1  # SE
    env[nw] = 2  # NW
    env[ne] = 3  # NE

    return env


def plot_bcd_runtime(runtime_dict, out_dir):
    def get_ci(vals):
        mean = np.mean(vals)
        stderr = np.std(vals, ddof=1) / np.sqrt(B)
        return mean, mean - 1.96 * stderr, mean + 1.96 * stderr

    baseline_key = f"{NAME_RF}(posthoc-mse)"
    base_vals = runtime_dict[baseline_key]
    base_mean, base_lo, base_hi = get_ci(base_vals)

    bcd_means, bcd_lowers, bcd_uppers = [], [], []
    for bs in BLOCK_SIZES:
        vals = runtime_dict[f"{NAME_RF}(posthoc-mse-BCD-{bs})"]
        mean, lo, hi = get_ci(vals)
        bcd_means.append(mean)
        bcd_lowers.append(lo)
        bcd_uppers.append(hi)

    plt.figure(figsize=(8, 5))
    plt.hlines(
        y=base_mean,
        xmin=5,
        xmax=50,
        linestyle="--",
        color="#F89C20",
        label="Non-BCD",
    )
    plt.fill_between(
        BLOCK_SIZES,
        [base_lo] * len(BLOCK_SIZES),
        [base_hi] * len(BLOCK_SIZES),
        color="#F89C20",
        alpha=0.2,
    )

    plt.plot(BLOCK_SIZES, bcd_means, color="#5790FC", marker="o", label="BCD")
    plt.fill_between(
        BLOCK_SIZES, bcd_lowers, bcd_uppers, color="#5790FC", alpha=0.3
    )

    plt.xlabel("Block size $b$")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_block_size.png"), dpi=300)
    plt.close()


def plot_mse_by_method(mse_envs_dict, out_dir):
    env_labels = ["Env 1", "Env 2", "Env 3", "Env 4"]
    env_colors = ["#5790FC", "#F89C20", "#964A8B", "#E42536"]

    methods = list(mse_envs_dict.keys())
    methods_labels = ["RF", "Non-BCD"] + [
        f"BCD ($b$={b})" for b in BLOCK_SIZES
    ]
    num_methods = len(methods)

    # For each method, per-environment mean MSE over B repetitions
    mse_matrix = np.array([np.mean(mse_envs_dict[m], axis=0) for m in methods])

    x = np.arange(num_methods)
    width = 0.2

    plt.figure(figsize=(15, 6))
    for env_idx in range(4):
        mse_vals = mse_matrix[:, env_idx]
        plt.bar(
            x + width * (env_idx - 1.5),
            mse_vals,
            width,
            label=env_labels[env_idx],
            color=env_colors[env_idx],
        )

    plt.xticks(x, methods_labels)
    plt.ylabel("Mean MSE per Environment")
    plt.xlabel("Method")
    plt.grid(True, axis="y", linewidth=0.3)
    plt.legend()
    plt.ylim(0.3, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bcd_mse_per_environment.png"), dpi=300)
    plt.close()


def plot_max_mse_vs_blocksize(maxmse_dict, out_dir):
    def get_ci(vals):
        mean = np.mean(vals)
        stderr = np.std(vals, ddof=1) / np.sqrt(B)
        return mean, mean - 1.96 * stderr, mean + 1.96 * stderr

    rf_mean, rf_lo, rf_hi = get_ci(maxmse_dict["RF"])
    posthoc_key = f"{NAME_RF}(posthoc-mse)"
    posthoc_mean, posthoc_lo, posthoc_hi = get_ci(maxmse_dict[posthoc_key])

    bcd_means = []
    bcd_lowers = []
    bcd_uppers = []
    for bs in BLOCK_SIZES:
        vals = maxmse_dict[f"{NAME_RF}(posthoc-mse-BCD-{bs})"]
        mean, lo, hi = get_ci(vals)
        bcd_means.append(mean)
        bcd_lowers.append(lo)
        bcd_uppers.append(hi)

    plt.figure(figsize=(8, 5))
    plt.hlines(
        y=rf_mean,
        xmin=5,
        xmax=50,
        linestyle="--",
        color="#5790FC",
        label="RF",
    )
    plt.fill_between(
        BLOCK_SIZES,
        [rf_lo] * len(BLOCK_SIZES),
        [rf_hi] * len(BLOCK_SIZES),
        color="#5790FC",
        alpha=0.2,
    )

    plt.hlines(
        y=posthoc_mean,
        xmin=5,
        xmax=50,
        linestyle="--",
        color="#F89C20",
        label=f"{NAME_RF}(posthoc-mse)",
    )
    plt.fill_between(
        BLOCK_SIZES,
        [posthoc_lo] * len(BLOCK_SIZES),
        [posthoc_hi] * len(BLOCK_SIZES),
        color="#F89C20",
        alpha=0.2,
    )

    plt.plot(
        BLOCK_SIZES,
        bcd_means,
        color="#964A8B",
        marker="o",
        label=f"{NAME_RF}(posthoc-mse-BCD)",
        markeredgecolor="white",
    )
    plt.fill_between(
        BLOCK_SIZES, bcd_lowers, bcd_uppers, color="#964A8B", alpha=0.3
    )

    plt.xlabel("Block size $b$")
    plt.ylabel("Max MSE across environments")
    plt.grid(True, linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bcd_max_mse.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    Z = X[["Latitude", "Longitude"]]
    X = X.drop(["Latitude", "Longitude"], axis=1)

    env = assign_quadrant(Z)

    methods = ["RF", f"{NAME_RF}(posthoc-mse)"] + [
        f"{NAME_RF}(posthoc-mse-BCD-{bs})" for bs in range(5, 55, 5)
    ]
    mse_envs_dict = {m: [] for m in methods}
    maxmse_dict = {m: [] for m in methods}
    runtime_dict = {m: [] for m in methods if m != "RF"}

    for b in range(B):
        print(f"\n### Repetition: {b+1}/{B} ###")
        # Stratified split (preserve env proportions among the 3 training envs)
        (
            X_tr,
            X_val,
            y_tr,
            y_val,
            env_tr,
            env_val,
        ) = train_test_split(
            X,
            y,
            env,
            test_size=VAL_PERCENTAGE,
            random_state=b,
            stratify=env,
        )

        print("Fitting standard Random Forest...", end=" ", flush=True)
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)
        print("Done!", flush=True)
        preds_rf = rf.predict(X_val)
        mse_envs_rf, maxmse_rf = max_mse(
            y_val, preds_rf, env_val, ret_ind=True
        )
        mse_envs_dict["RF"].append(mse_envs_rf)
        maxmse_dict["RF"].append(maxmse_rf)

        # Non-BCD
        rf_nonbcd = copy.deepcopy(rf)
        print("Modifying predictions with non-BCD...", end=" ", flush=True)
        start = time.time()
        rf_nonbcd.modify_predictions_trees(env_tr, solver="ECOS")
        end = time.time()
        print("Done!", flush=True)
        non_bcd_time = end - start
        preds_nonbcd = rf_nonbcd.predict(X_val)
        mse_envs_nonbcd, maxmse_nonbcd = max_mse(
            y_val, preds_nonbcd, env_val, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(posthoc-mse)"].append(mse_envs_nonbcd)
        maxmse_dict[f"{NAME_RF}(posthoc-mse)"].append(maxmse_nonbcd)
        runtime_dict[f"{NAME_RF}(posthoc-mse)"].append(non_bcd_time)

        # BCD Variants
        for bs in BLOCK_SIZES:
            rf_bcd = copy.deepcopy(rf)
            print(
                f"Modifying predictions with BCD (bs {bs})...",
                end=" ",
                flush=True,
            )
            start = time.time()
            rf_bcd.modify_predictions_trees(
                env_tr, bcd=True, patience=1, block_size=bs, solver="ECOS"
            )
            end = time.time()
            print("Done!", flush=True)
            bcd_time = end - start
            preds_bcd = rf_bcd.predict(X_val)
            mse_envs_bcd, maxmse_bcd = max_mse(
                y_val, preds_bcd, env_val, ret_ind=True
            )
            mse_envs_dict[f"{NAME_RF}(posthoc-mse-BCD-{bs})"].append(
                mse_envs_bcd
            )
            maxmse_dict[f"{NAME_RF}(posthoc-mse-BCD-{bs})"].append(maxmse_bcd)
            runtime_dict[f"{NAME_RF}(posthoc-mse-BCD-{bs})"].append(bcd_time)

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(parent_dir, "results")
    plots_dir = os.path.join(results_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)

    plot_bcd_runtime(runtime_dict, plots_dir)
    plot_mse_by_method(mse_envs_dict, plots_dir)
    plot_max_mse_vs_blocksize(maxmse_dict, plots_dir)

    print(f"\nSaved plots to {plots_dir}")
