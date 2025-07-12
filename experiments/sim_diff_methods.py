import os
import copy
import time
from sklearn.metrics import mean_squared_error
from nldg.utils import *
from nldg.rf import MaggingRF
from adaXT.random_forest import RandomForest
from tqdm import tqdm
from utils import *

N_SIM = 20
SAMPLE_SIZE = 1000
NOISE_STD = 0.5
N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SIM_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(SIM_DIR, exist_ok=True)
OUT_DIR = os.path.join(SIM_DIR, "sim_diff_methods")
os.makedirs(OUT_DIR, exist_ok=True)

NAME_RF = "WORME-RF"


if __name__ == "__main__":
    results_dict = {
        "RF": [],
        "RF(magging)": [],
        f"{NAME_RF}(local)": [],
        f"{NAME_RF}(posthoc)": [],
        f"{NAME_RF}(posthoc-local)": [],
        f"{NAME_RF}(global-dfs)": [],
        f"{NAME_RF}(global)": [],
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

        # Default RF
        start = time.process_time()
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
        )
        rf.fit(Xtr, Ytr)
        end = time.process_time()
        time_rf = end - start
        runtime_dict["RF"].append(time_rf)
        fitted_rf = rf.predict(Xtr)
        mse_dict["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mse_envs_rf, maxmse_rf = max_mse(Ytr, fitted_rf, Etr, ret_ind=True)
        mse_envs_dict["RF"].append(mse_envs_rf)
        maxmse_dict["RF"].append(maxmse_rf)

        # Magging
        start = time.process_time()
        rf_magging = MaggingRF(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=i,
            backend="adaXT",
        )
        fitted_magging, preds_magging = rf_magging.fit_predict_magging(
            Xtr, Ytr, Etr, Xtr
        )
        end = time.process_time()
        runtime_dict["RF(magging)"].append(end - start)
        mse_dict["RF(magging)"].append(mean_squared_error(Ytr, fitted_magging))
        mse_envs_magging, maxmse_magging = max_mse(
            Ytr, fitted_magging, Etr, ret_ind=True
        )
        mse_envs_dict["RF(magging)"].append(mse_envs_magging)
        maxmse_dict["RF(magging)"].append(maxmse_magging)

        # Local
        start = time.process_time()
        rf_minmax_m0 = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="base",
        )
        rf_minmax_m0.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        time_minmax_m0 = end - start
        runtime_dict[f"{NAME_RF}(local)"].append(time_minmax_m0)
        fitted_minmax_m0 = rf_minmax_m0.predict(Xtr)
        mse_dict[f"{NAME_RF}(local)"].append(
            mean_squared_error(Ytr, fitted_minmax_m0)
        )
        mse_envs_minmax_m0, maxmse_minmax_m0 = max_mse(
            Ytr, fitted_minmax_m0, Etr, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(local)"].append(mse_envs_minmax_m0)
        maxmse_dict[f"{NAME_RF}(local)"].append(maxmse_minmax_m0)

        # Post-hoc
        start = time.process_time()
        rf.modify_predictions_trees(Etr)
        end = time.process_time()
        time_minmax_m1 = end - start
        time_minmax_m1 += time_rf
        runtime_dict[f"{NAME_RF}(posthoc)"].append(time_minmax_m1)
        fitted_minmax_m1 = rf.predict(Xtr)
        mse_dict[f"{NAME_RF}(posthoc)"].append(
            mean_squared_error(Ytr, fitted_minmax_m1)
        )
        mse_envs_minmax_m1, maxmse_minmax_m1 = max_mse(
            Ytr, fitted_minmax_m1, Etr, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(posthoc)"].append(mse_envs_minmax_m1)
        maxmse_dict[f"{NAME_RF}(posthoc)"].append(maxmse_minmax_m1)

        # Posthoc to Local
        start = time.process_time()
        rf_minmax_m0.modify_predictions_trees(Etr)
        end = time.process_time()
        time_minmax_m2 = end - start
        time_minmax_m2 += time_minmax_m0
        runtime_dict[f"{NAME_RF}(posthoc-local)"].append(time_minmax_m2)
        fitted_minmax_m2 = rf_minmax_m0.predict(Xtr)
        mse_dict[f"{NAME_RF}(posthoc-local)"].append(
            mean_squared_error(Ytr, fitted_minmax_m2)
        )
        mse_envs_minmax_m2, maxmse_minmax_m2 = max_mse(
            Ytr, fitted_minmax_m2, Etr, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(posthoc-local)"].append(mse_envs_minmax_m2)
        maxmse_dict[f"{NAME_RF}(posthoc-local)"].append(maxmse_minmax_m2)

        # Global - DFS
        start = time.process_time()
        rf_minmax_m3 = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="fullopt",
        )
        rf_minmax_m3.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        runtime_dict[f"{NAME_RF}(global-dfs)"].append(end - start)
        fitted_minmax_m3 = rf_minmax_m3.predict(Xtr)
        mse_dict[f"{NAME_RF}(global-dfs)"].append(
            mean_squared_error(Ytr, fitted_minmax_m3)
        )
        mse_envs_minmax_m3, maxmse_minmax_m3 = max_mse(
            Ytr, fitted_minmax_m3, Etr, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(global-dfs)"].append(mse_envs_minmax_m3)
        maxmse_dict[f"{NAME_RF}(global-dfs)"].append(maxmse_minmax_m3)

        # Global
        start = time.process_time()
        rf_minmax_m4 = RandomForest(
            "MinMaxRegression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=i,
            minmax_method="adafullopt",
        )
        rf_minmax_m4.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        runtime_dict[f"{NAME_RF}(global)"].append(end - start)
        fitted_minmax_m4 = rf_minmax_m4.predict(Xtr)
        mse_dict[f"{NAME_RF}(global)"].append(
            mean_squared_error(Ytr, fitted_minmax_m4)
        )
        mse_envs_minmax_m4, maxmse_minmax_m4 = max_mse(
            Ytr, fitted_minmax_m4, Etr, ret_ind=True
        )
        mse_envs_dict[f"{NAME_RF}(global)"].append(mse_envs_minmax_m4)
        maxmse_dict[f"{NAME_RF}(global)"].append(maxmse_minmax_m4)

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
            f.write("MSE\n")
            for col in mse_df.columns:
                f.write(
                    f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
                )
            f.write("\n")

            # Max MSE
            means = maxmse_df.mean(axis=0)
            n = maxmse_df.shape[0]
            stderr = maxmse_df.std(axis=0, ddof=1) / np.sqrt(n)
            ci_lower = means - 1.96 * stderr
            ci_upper = means + 1.96 * stderr
            f.write("Max MSE\n")
            for col in maxmse_df.columns:
                f.write(
                    f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
                )
            f.write("\n")

            # Runtime
            means = runtime_df.mean(axis=0)
            n = runtime_df.shape[0]
            stderr = runtime_df.std(axis=0, ddof=1) / np.sqrt(n)
            ci_lower = means - 1.96 * stderr
            ci_upper = means + 1.96 * stderr
            f.write("Runtime (seconds)\n")
            for col in runtime_df.columns:
                f.write(
                    f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
                )

        plot_max_mse_boxplot(maxmse_df, saveplot=True)
