import os
import copy
import time
from sklearn.metrics import mean_squared_error
from nldg.utils import *
from adaXT.random_forest import RandomForest
from tqdm import tqdm
from utils import *

N_SIM = 20
SAMPLE_SIZE = 1000
NOISE_STD = 0.5
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 15
N_JOBS = 10
SEED = 42
BLOCK_SIZE = 15
PATIENCE = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SIM_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(SIM_DIR, exist_ok=True)
OUT_DIR = os.path.join(SIM_DIR, "sim_eg_bcd")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_dtr(
    dtr: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "comparison_eg_bcd",
):
    data_colors = ["black", "grey", "silver"]
    environments = sorted(dtr["E"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, env in enumerate(environments):
        marker_style = "o"
        ax.scatter(
            dtr[dtr["E"] == env]["X"],
            dtr[dtr["E"] == env]["Y"],
            color=data_colors[idx],
            marker=marker_style,
            alpha=0.5,
            s=30,
            label=f"Env {env + 1}",
        )
    ax.plot(
        dtr["X_sorted"],
        dtr["rf"],
        color="#5790FC",
        linewidth=2,
        label="RF",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["rf-posthoc"],
        color="#F89C20",
        linewidth=2,
        label="MaxRM-RF(mse)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["rf-posthoc-eg"],
        color="#86C8DD",
        linewidth=2,
        label="MaxRM-RF-EG(mse)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["rf-posthoc-bcd"],
        color="#964A8B",
        linewidth=2,
        label="MaxRM-RF-BCD(mse)",
    )

    x_range = np.linspace(dtr["X_sorted"].min(), dtr["X_sorted"].max(), 1000)
    y_opt = np.where(x_range > 0, 2.25 * x_range, 1.25 * x_range)
    ax.plot(
        x_range,
        y_opt,
        color="#E42536",
        linewidth=3,
        label="Oracle",
        linestyle="--",
    )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles)

    plt.tight_layout()
    if saveplot:
        outpath = os.path.join(OUT_DIR, f"{nameplot}.pdf")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # plot
    dtr = gen_data_v6(
        n=SAMPLE_SIZE, noise_std=NOISE_STD, random_state=SEED, setting=2
    )
    Xtr = np.array(dtr.drop(columns=["E", "Y"]))
    Ytr = np.array(dtr["Y"])
    Etr = np.array(dtr["E"])
    Xtr_sorted = np.sort(Xtr, axis=0)

    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf.fit(Xtr, Ytr)
    preds_rf = rf.predict(Xtr_sorted)

    rf_posthoc = copy.deepcopy(rf)
    rf_posthoc.modify_predictions_trees(Etr)
    preds_posthoc = rf_posthoc.predict(Xtr_sorted)

    rf_posthoc_eg = copy.deepcopy(rf)
    rf_posthoc_eg.modify_predictions_trees(
        Etr, opt_method="extragradient", early_stopping=True
    )
    preds_posthoc_eg = rf_posthoc_eg.predict(Xtr_sorted)

    rf_posthoc_bcd = copy.deepcopy(rf)
    block_size = 5
    rf_posthoc_bcd.modify_predictions_trees(
        Etr, bcd=True, block_size=BLOCK_SIZE, patience=PATIENCE
    )
    preds_posthoc_bcd = rf_posthoc_bcd.predict(Xtr_sorted)

    dtr["X_sorted"] = Xtr_sorted
    dtr["rf"] = preds_rf
    dtr["rf-posthoc"] = preds_posthoc
    dtr["rf-posthoc-eg"] = preds_posthoc_eg
    dtr["rf-posthoc-bcd"] = preds_posthoc_bcd
    plot_dtr(dtr, saveplot=True)

    # simulation
    results_dict = {
        "RF": [],
        "MaxRM-RF-posthoc": [],
        "MaxRM-RF-posthoc-EG": [],
        "MaxRM-RF-posthoc-BCD": [],
    }

    runtime_dict = copy.deepcopy(results_dict)
    mse_envs_dict = copy.deepcopy(results_dict)
    maxmse_dict = copy.deepcopy(results_dict)
    mse_dict = copy.deepcopy(results_dict)

    for i in tqdm(range(N_SIM)):
        dtr = gen_data_v6(
            n=SAMPLE_SIZE,
            noise_std=NOISE_STD,
            random_state=SEED + i,
            setting=2,
        )
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        dte = gen_data_v6(
            n=SAMPLE_SIZE,
            noise_std=NOISE_STD,
            random_state=1000 + SEED + i,
            setting=2,
        )
        Xte = np.array(dte.drop(columns=["E", "Y"]))
        Yte = np.array(dte["Y"])
        Ete = np.array(dte["E"])

        # RF
        start = time.perf_counter()
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Ytr)
        end = time.perf_counter()
        time_rf = end - start
        runtime_dict["RF"].append(time_rf)
        preds_rf = rf.predict(Xte)
        mse_dict["RF"].append(mean_squared_error(Yte, preds_rf))
        mse_envs_rf, maxmse_rf = max_mse(Yte, preds_rf, Ete, ret_ind=True)
        mse_envs_dict["RF"].append(mse_envs_rf)
        maxmse_dict["RF"].append(maxmse_rf)

        # MaxRM-RF-posthoc
        rf_posthoc = copy.deepcopy(rf)
        start = time.perf_counter()
        rf_posthoc.modify_predictions_trees(
            Etr,
            n_jobs=N_JOBS,
        )
        end = time.perf_counter()
        time_posthoc = end - start
        time_posthoc += time_rf
        runtime_dict["MaxRM-RF-posthoc"].append(time_posthoc)
        preds_posthoc = rf_posthoc.predict(Xte)
        mse_dict["MaxRM-RF-posthoc"].append(
            mean_squared_error(Yte, preds_posthoc)
        )
        mse_envs_posthoc, maxmse_posthoc = max_mse(
            Yte, preds_posthoc, Ete, ret_ind=True
        )
        mse_envs_dict["MaxRM-RF-posthoc"].append(mse_envs_posthoc)
        maxmse_dict["MaxRM-RF-posthoc"].append(maxmse_posthoc)

        # MaxRM-RF-posthoc-EG
        rf_posthoc_eg = copy.deepcopy(rf)
        start = time.perf_counter()
        rf_posthoc_eg.modify_predictions_trees(
            Etr,
            opt_method="extragradient",
            early_stopping=True,
            n_jobs=N_JOBS,
        )
        end = time.perf_counter()
        time_posthoc_eg = end - start
        time_posthoc_eg += time_rf
        runtime_dict["MaxRM-RF-posthoc-EG"].append(time_posthoc_eg)
        preds_posthoc_eg = rf_posthoc_eg.predict(Xte)
        mse_dict["MaxRM-RF-posthoc-EG"].append(
            mean_squared_error(Yte, preds_posthoc_eg)
        )
        mse_envs_posthoc_eg, maxmse_posthoc_eg = max_mse(
            Yte, preds_posthoc_eg, Ete, ret_ind=True
        )
        mse_envs_dict["MaxRM-RF-posthoc-EG"].append(mse_envs_posthoc_eg)
        maxmse_dict["MaxRM-RF-posthoc-EG"].append(maxmse_posthoc_eg)

        # MaxRM-RF-posthoc-BCD
        rf_posthoc_bcd = copy.deepcopy(rf)
        start = time.perf_counter()
        rf_posthoc_bcd.modify_predictions_trees(
            Etr,
            bcd=True,
            block_size=BLOCK_SIZE,
            patience=PATIENCE,
            n_jobs=N_JOBS,
        )
        end = time.perf_counter()
        time_posthoc_bcd = end - start
        time_posthoc_bcd += time_rf
        runtime_dict["MaxRM-RF-posthoc-BCD"].append(time_posthoc_bcd)
        preds_posthoc_bcd = rf_posthoc_bcd.predict(Xte)
        mse_dict["MaxRM-RF-posthoc-BCD"].append(
            mean_squared_error(Yte, preds_posthoc_bcd)
        )
        mse_envs_posthoc_bcd, maxmse_posthoc_bcd = max_mse(
            Yte, preds_posthoc_bcd, Ete, ret_ind=True
        )
        mse_envs_dict["MaxRM-RF-posthoc-BCD"].append(mse_envs_posthoc_bcd)
        maxmse_dict["MaxRM-RF-posthoc-BCD"].append(maxmse_posthoc_bcd)

    # Results
    mse_df = pd.DataFrame(mse_dict)
    mse_envs_df = get_df(mse_envs_dict)
    maxmse_df = pd.DataFrame(maxmse_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    output_path = os.path.join(OUT_DIR, "summary_eg_bcd.txt")
    with open(output_path, "w") as f:
        # MSE
        means = mse_df.mean(axis=0)
        n = mse_df.shape[0]
        stderr = mse_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        half_width = (ci_upper - ci_lower) / 2
        f.write("MSE\n")
        for col in mse_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], "
                f"CI half-width = {half_width[col]:.4f}\n"
            )
        f.write("\n")

        # Max MSE
        means = maxmse_df.mean(axis=0)
        n = maxmse_df.shape[0]
        stderr = maxmse_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        half_width = (ci_upper - ci_lower) / 2
        f.write("Max MSE\n")
        for col in maxmse_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], "
                f"CI half-width = {half_width[col]:.4f}\n"
            )
        f.write("\n")

        # Runtime
        means = runtime_df.mean(axis=0)
        n = runtime_df.shape[0]
        stderr = runtime_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        f.write("Runtime (seconds)\n")
        half_width = (ci_upper - ci_lower) / 2
        for col in runtime_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], "
                f"CI half-width = {half_width[col]:.4f}\n"
            )

    # plot_max_mse_boxplot(maxmse_df, saveplot=True, out_dir=OUT_DIR)

    print(f"Saved results to {OUT_DIR}")
