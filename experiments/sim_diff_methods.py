import os
import copy
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from nldg.utils import *
from adaXT.random_forest import RandomForest
from tqdm import tqdm
from utils import *

N_SIM = 20
SAMPLE_SIZE = 1000
NOISE_STD = 0.5
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 15
N_JOBS = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SIM_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(SIM_DIR, exist_ok=True)
OUT_DIR = os.path.join(SIM_DIR, "sim_diff_methods")
os.makedirs(OUT_DIR, exist_ok=True)

NAME_RF = "MaxRM-RF"


if __name__ == "__main__":
    results_dict = {
        f"{NAME_RF}-posthoc": [],
        f"{NAME_RF}-local": [],
        f"{NAME_RF}-global": [],
        f"{NAME_RF}-global-NonDFS": [],
        f"{NAME_RF}-ow": [],
        f"{NAME_RF}-posthoc-ow": [],
        f"{NAME_RF}-local-ow": [],
        f"{NAME_RF}-global-ow": [],
        f"{NAME_RF}-global-NonDFS-ow": [],
        "RF": [],
    }

    runtime_dict = copy.deepcopy(results_dict)
    mse_envs_dict = copy.deepcopy(results_dict)
    maxmse_dict = copy.deepcopy(results_dict)
    mse_dict = copy.deepcopy(results_dict)

    for i in tqdm(range(N_SIM)):
        dtr = gen_data_v6(
            n=SAMPLE_SIZE, noise_std=NOISE_STD, random_state=i, setting=2
        )
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        dte = gen_data_v6(
            n=SAMPLE_SIZE,
            noise_std=NOISE_STD,
            random_state=1000 + i,
            setting=2,
        )
        Xte = np.array(dte.drop(columns=["E", "Y"]))
        Yte = np.array(dte["Y"])
        Ete = np.array(dte["E"])

        # Default RF ------------------------------------------------
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
        # -----------------------------------------------------------

        # MaxRM-RF-posthoc ------------------------------------------
        start = time.perf_counter()
        rf.modify_predictions_trees(
            Etr,
            n_jobs=N_JOBS,
        )
        end = time.perf_counter()
        time_posthoc = end - start
        time_posthoc += time_rf
        runtime_dict[f"{NAME_RF}-posthoc"].append(time_posthoc)
        preds_posthoc = rf.predict(Xte)
        mse_dict[f"{NAME_RF}-posthoc"].append(
            mean_squared_error(Yte, preds_posthoc)
        )
        mse_envs_posthoc, maxmse_posthoc = max_mse(
            Yte, preds_posthoc, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-posthoc"].append(mse_envs_posthoc)
        maxmse_dict[f"{NAME_RF}-posthoc"].append(maxmse_posthoc)
        # -----------------------------------------------------------

        # MaxRM-RF-local --------------------------------------------
        start = time.perf_counter()
        rf_local = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="base",
            n_jobs=N_JOBS,
        )
        rf_local.fit(Xtr, Ytr, Etr)
        end = time.perf_counter()
        time_local = end - start
        runtime_dict[f"{NAME_RF}-local"].append(time_local)
        preds_local = rf_local.predict(Xte)
        mse_dict[f"{NAME_RF}-local"].append(
            mean_squared_error(Yte, preds_local)
        )
        mse_envs_local, maxmse_local = max_mse(
            Yte, preds_local, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-local"].append(mse_envs_local)
        maxmse_dict[f"{NAME_RF}-local"].append(maxmse_local)
        # -----------------------------------------------------------

        # MaxRM-RF-global -------------------------------------------
        start = time.perf_counter()
        rf_global = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="fullopt",
            n_jobs=N_JOBS,
        )
        rf_global.fit(Xtr, Ytr, Etr)
        end = time.perf_counter()
        runtime_dict[f"{NAME_RF}-global"].append(end - start)
        preds_global = rf_global.predict(Xte)
        mse_dict[f"{NAME_RF}-global"].append(
            mean_squared_error(Yte, preds_global)
        )
        mse_envs_global, maxmse_global = max_mse(
            Yte, preds_global, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-global"].append(mse_envs_global)
        maxmse_dict[f"{NAME_RF}-global"].append(maxmse_global)
        # -----------------------------------------------------------

        # MaxRM-RF-global-NonDFS ------------------------------------
        start = time.perf_counter()
        rf_global_nondfs = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="adafullopt",
            n_jobs=N_JOBS,
        )
        rf_global_nondfs.fit(Xtr, Ytr, Etr)
        end = time.perf_counter()
        runtime_dict[f"{NAME_RF}-global-NonDFS"].append(end - start)
        preds_global_nondfs = rf_global_nondfs.predict(Xte)
        mse_dict[f"{NAME_RF}-global-NonDFS"].append(
            mean_squared_error(Yte, preds_global_nondfs)
        )
        mse_envs_global_nondfs, maxmse_global_nondfs = max_mse(
            Yte, preds_global_nondfs, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-global-NonDFS"].append(
            mse_envs_global_nondfs
        )
        maxmse_dict[f"{NAME_RF}-global-NonDFS"].append(maxmse_global_nondfs)
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # optimally-weighted versions
        # -----------------------------------------------------------

        # split train into a part for tree-fitting and a part for weight-refinement
        Xtr_t, Xtr_w, Ytr_t, Ytr_w, Etr_t, Etr_w = train_test_split(
            Xtr, Ytr, Etr, test_size=0.3, random_state=i, shuffle=True
        )

        start_rf_t = time.perf_counter()
        rf_t = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            n_jobs=N_JOBS,
        )
        rf_t.fit(Xtr_t, Ytr_t)
        end_rf_t = time.perf_counter()
        time_rf_t = end_rf_t - start_rf_t

        # MaxRM-RF-ow -----------------------------------------------
        start = time.perf_counter()
        preds_ow, _ = rf_t.refine_weights(
            X_val=Xtr_w,
            Y_val=Ytr_w,
            E_val=Etr_w,
            X=Xte,
        )
        end = time.perf_counter()

        time_MaxRM_RF_ow = time_rf_t + end - start
        runtime_dict[f"{NAME_RF}-ow"].append(time_MaxRM_RF_ow)
        mse_dict[f"{NAME_RF}-ow"].append(mean_squared_error(Yte, preds_ow))
        mse_envs_ow, maxmse_ow = max_mse(Yte, preds_ow, Ete, ret_ind=True)
        mse_envs_dict[f"{NAME_RF}-ow"].append(mse_envs_ow)
        maxmse_dict[f"{NAME_RF}-ow"].append(maxmse_ow)
        # -----------------------------------------------------------

        # MaxRM-RF-posthoc-ow ---------------------------------------
        start = time.perf_counter()
        rf_t.modify_predictions_trees(
            Etr_t,
            n_jobs=N_JOBS,
        )
        preds_posthoc_ow, _ = rf_t.refine_weights(
            X_val=Xtr_w,
            Y_val=Ytr_w,
            E_val=Etr_w,
            X=Xte,
        )
        end = time.perf_counter()
        time_posthoc_ow = time_rf_t + end - start
        runtime_dict[f"{NAME_RF}-posthoc-ow"].append(time_posthoc_ow)
        mse_dict[f"{NAME_RF}-posthoc-ow"].append(
            mean_squared_error(Yte, preds_posthoc_ow)
        )
        mse_envs_posthoc_ow, maxmse_posthoc_ow = max_mse(
            Yte, preds_posthoc_ow, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-posthoc-ow"].append(mse_envs_posthoc_ow)
        maxmse_dict[f"{NAME_RF}-posthoc-ow"].append(maxmse_posthoc_ow)
        # -----------------------------------------------------------

        # MaxRM-RF-local-ow -----------------------------------------
        start = time.perf_counter()
        rf_local_t = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="base",
            n_jobs=N_JOBS,
        )
        rf_local_t.fit(Xtr_t, Ytr_t, Etr_t)
        preds_local_ow, _ = rf_local_t.refine_weights(
            X_val=Xtr_w,
            Y_val=Ytr_w,
            E_val=Etr_w,
            X=Xte,
        )
        end = time.perf_counter()
        time_local_ow = end - start
        runtime_dict[f"{NAME_RF}-local-ow"].append(time_local_ow)
        mse_dict[f"{NAME_RF}-local-ow"].append(
            mean_squared_error(Yte, preds_local_ow)
        )
        mse_envs_local_ow, maxmse_local_ow = max_mse(
            Yte, preds_local_ow, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-local-ow"].append(mse_envs_local_ow)
        maxmse_dict[f"{NAME_RF}-local-ow"].append(maxmse_local_ow)
        # -----------------------------------------------------------

        # MaxRM-RF-global-ow ----------------------------------------
        start = time.perf_counter()
        rf_global_t = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="fullopt",
            n_jobs=N_JOBS,
        )
        rf_global_t.fit(Xtr_t, Ytr_t, Etr_t)
        preds_global_ow, _ = rf_global_t.refine_weights(
            X_val=Xtr_w,
            Y_val=Ytr_w,
            E_val=Etr_w,
            X=Xte,
        )
        end = time.perf_counter()
        time_global_ow = end - start
        runtime_dict[f"{NAME_RF}-global-ow"].append(time_global_ow)
        mse_dict[f"{NAME_RF}-global-ow"].append(
            mean_squared_error(Yte, preds_global_ow)
        )
        mse_envs_global_ow, maxmse_global_ow = max_mse(
            Yte, preds_global_ow, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-global-ow"].append(mse_envs_global_ow)
        maxmse_dict[f"{NAME_RF}-global-ow"].append(maxmse_global_ow)
        # -----------------------------------------------------------

        # MaxRM-RF-global-NonDFS-ow ---------------------------------
        start = time.perf_counter()
        rf_global_nondfs_t = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="adafullopt",
            n_jobs=N_JOBS,
        )
        rf_global_nondfs_t.fit(Xtr_t, Ytr_t, Etr_t)
        preds_global_nondfs_ow, _ = rf_global_nondfs_t.refine_weights(
            X_val=Xtr_w,
            Y_val=Ytr_w,
            E_val=Etr_w,
            X=Xte,
        )
        end = time.perf_counter()
        time_global_nondfs_ow = end - start
        runtime_dict[f"{NAME_RF}-global-NonDFS-ow"].append(
            time_global_nondfs_ow
        )
        mse_dict[f"{NAME_RF}-global-NonDFS-ow"].append(
            mean_squared_error(Yte, preds_global_nondfs_ow)
        )
        mse_envs_global_nondfs_ow, maxmse_global_nondfs_ow = max_mse(
            Yte, preds_global_nondfs_ow, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}-global-NonDFS-ow"].append(
            mse_envs_global_nondfs_ow
        )
        maxmse_dict[f"{NAME_RF}-global-NonDFS-ow"].append(
            maxmse_global_nondfs_ow
        )
        # -----------------------------------------------------------

    # Results
    mse_df = pd.DataFrame(mse_dict)
    mse_envs_df = get_df(mse_envs_dict)
    maxmse_df = pd.DataFrame(maxmse_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    output_path = os.path.join(OUT_DIR, "summary_all.txt")
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
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], CI half-width = {half_width[col]:.4f}\n"
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
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], CI half-width = {half_width[col]:.4f}\n"
            )
        f.write("\n")

        # Runtime
        means = runtime_df.mean(axis=0)
        n = runtime_df.shape[0]
        stderr = runtime_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        half_width = (ci_upper - ci_lower) / 2
        f.write("Runtime (seconds)\n")
        for col in runtime_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}], CI half-width = {half_width[col]:.4f}\n"
            )

    # plot_max_mse_boxplot(maxmse_df, saveplot=True, out_dir=OUT_DIR)

    print(f"Saved results to {OUT_DIR}")
