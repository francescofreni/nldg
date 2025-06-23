import argparse
import os
import copy
import time
from sklearn.metrics import mean_squared_error
from nldg.utils import *
from nldg.rf import MaggingRF_PB
from adaXT.random_forest import RandomForest
from tqdm import tqdm
from utils import plot_max_mse_msl, plot_max_mse_boxplot


def main(
    nsim: int,
    n: int,
    noise_std: float,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
):
    # First simulation
    results_dict = {
        "RF": [],
        "MaggingRF": [],
        "L-MMRF": [],
        "Post-RF": [],
        "Post-L-MMRF": [],
        "G-DFS-MMRF": [],
        "G-MMRF": [],
    }

    runtime_dict = copy.deepcopy(results_dict)
    mse_envs_dict = copy.deepcopy(results_dict)
    maxmse_dict = copy.deepcopy(results_dict)
    mse_dict = copy.deepcopy(results_dict)

    for i in tqdm(range(nsim)):
        dtr = gen_data_v6(n=n, noise_std=noise_std, random_state=i)
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        # Default RF
        start = time.process_time()
        rf = RandomForest(
            "Regression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
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

        # MaggingRF
        start = time.process_time()
        rf_magging = MaggingRF_PB(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=i,
            backend="adaXT",
        )
        fitted_magging, preds_magging = rf_magging.fit_predict_magging(
            Xtr, Ytr, Etr, Xtr
        )
        end = time.process_time()
        runtime_dict["MaggingRF"].append(end - start)
        mse_dict["MaggingRF"].append(mean_squared_error(Ytr, fitted_magging))
        mse_envs_magging, maxmse_magging = max_mse(
            Ytr, fitted_magging, Etr, ret_ind=True
        )
        mse_envs_dict["MaggingRF"].append(mse_envs_magging)
        maxmse_dict["MaggingRF"].append(maxmse_magging)

        # MinMaxRF-M0 / L-MMRF
        start = time.process_time()
        rf_minmax_m0 = RandomForest(
            "MinMaxRegression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
            minmax_method="base",
        )
        rf_minmax_m0.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        time_minmax_m0 = end - start
        runtime_dict["L-MMRF"].append(time_minmax_m0)
        fitted_minmax_m0 = rf_minmax_m0.predict(Xtr)
        mse_dict["L-MMRF"].append(mean_squared_error(Ytr, fitted_minmax_m0))
        mse_envs_minmax_m0, maxmse_minmax_m0 = max_mse(
            Ytr, fitted_minmax_m0, Etr, ret_ind=True
        )
        mse_envs_dict["L-MMRF"].append(mse_envs_minmax_m0)
        maxmse_dict["L-MMRF"].append(maxmse_minmax_m0)

        # MinMaxRF-M1 / Post-RF
        start = time.process_time()
        rf.modify_predictions_trees(Etr)
        end = time.process_time()
        time_minmax_m1 = end - start
        time_minmax_m1 += time_rf
        runtime_dict["Post-RF"].append(time_minmax_m1)
        fitted_minmax_m1 = rf.predict(Xtr)
        mse_dict["Post-RF"].append(mean_squared_error(Ytr, fitted_minmax_m1))
        mse_envs_minmax_m1, maxmse_minmax_m1 = max_mse(
            Ytr, fitted_minmax_m1, Etr, ret_ind=True
        )
        mse_envs_dict["Post-RF"].append(mse_envs_minmax_m1)
        maxmse_dict["Post-RF"].append(maxmse_minmax_m1)

        # MinMaxRF-M2 / Post-L-MMRF
        start = time.process_time()
        rf_minmax_m0.modify_predictions_trees(Etr)
        end = time.process_time()
        time_minmax_m2 = end - start
        time_minmax_m2 += time_minmax_m0
        runtime_dict["Post-L-MMRF"].append(time_minmax_m2)
        fitted_minmax_m2 = rf_minmax_m0.predict(Xtr)
        mse_dict["Post-L-MMRF"].append(
            mean_squared_error(Ytr, fitted_minmax_m2)
        )
        mse_envs_minmax_m2, maxmse_minmax_m2 = max_mse(
            Ytr, fitted_minmax_m2, Etr, ret_ind=True
        )
        mse_envs_dict["Post-L-MMRF"].append(mse_envs_minmax_m2)
        maxmse_dict["Post-L-MMRF"].append(maxmse_minmax_m2)

        # MinMaxRF-M3 / G-DFS-MMRF
        start = time.process_time()
        rf_minmax_m3 = RandomForest(
            "MinMaxRegression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
            minmax_method="fullopt",
        )
        rf_minmax_m3.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        runtime_dict["G-DFS-MMRF"].append(end - start)
        fitted_minmax_m3 = rf_minmax_m3.predict(Xtr)
        mse_dict["G-DFS-MMRF"].append(
            mean_squared_error(Ytr, fitted_minmax_m3)
        )
        mse_envs_minmax_m3, maxmse_minmax_m3 = max_mse(
            Ytr, fitted_minmax_m3, Etr, ret_ind=True
        )
        mse_envs_dict["G-DFS-MMRF"].append(mse_envs_minmax_m3)
        maxmse_dict["G-DFS-MMRF"].append(maxmse_minmax_m3)

        # MinMaxRF-M4 / G-MMRF
        start = time.process_time()
        rf_minmax_m4 = RandomForest(
            "MinMaxRegression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
            minmax_method="adafullopt",
        )
        rf_minmax_m4.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        runtime_dict["G-MMRF"].append(end - start)
        fitted_minmax_m4 = rf_minmax_m4.predict(Xtr)
        mse_dict["G-MMRF"].append(mean_squared_error(Ytr, fitted_minmax_m4))
        mse_envs_minmax_m4, maxmse_minmax_m4 = max_mse(
            Ytr, fitted_minmax_m4, Etr, ret_ind=True
        )
        mse_envs_dict["G-MMRF"].append(mse_envs_minmax_m4)
        maxmse_dict["G-MMRF"].append(maxmse_minmax_m4)

    def get_df(res_dict):
        rows = []
        for method, lists in res_dict.items():
            for sim_id, res_envs in enumerate(lists):
                for env_id, res in enumerate(res_envs):
                    rows.append(
                        {
                            "MSE": res,
                            "env_id": env_id,
                            "sim_id": sim_id,
                            "method": method,
                        }
                    )
        return pd.DataFrame(rows)

    mse_df = pd.DataFrame(mse_dict)
    mse_envs_df = get_df(mse_envs_dict)
    maxmse_df = pd.DataFrame(maxmse_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(parent_dir, results_folder)
    os.makedirs(results_dir, exist_ok=True)
    out_data_dir = os.path.join(results_dir, "output_data_simulation_rf")
    os.makedirs(out_data_dir, exist_ok=True)

    mse_df.to_csv(os.path.join(out_data_dir, "mse.csv"), index=False)
    means = mse_df.mean(axis=0)
    n = mse_df.shape[0]
    stderr = mse_df.std(axis=0, ddof=1) / np.sqrt(n)
    ci_lower = means - 1.96 * stderr
    ci_upper = means + 1.96 * stderr
    summary_mse_df = pd.DataFrame(
        {"MSE": means, "CI Lower": ci_lower, "CI Upper": ci_upper}
    )
    summary_mse_df.to_csv(os.path.join(out_data_dir, "summary_mse.txt"))

    mse_envs_df.to_csv(os.path.join(out_data_dir, "mse_envs.csv"), index=False)

    maxmse_df.to_csv(os.path.join(out_data_dir, "max_mse.csv"), index=False)
    means = maxmse_df.mean(axis=0)
    n = maxmse_df.shape[0]
    stderr = maxmse_df.std(axis=0, ddof=1) / np.sqrt(n)
    ci_lower = means - 1.96 * stderr
    ci_upper = means + 1.96 * stderr
    summary_maxmse_df = pd.DataFrame(
        {"MSE": means, "CI Lower": ci_lower, "CI Upper": ci_upper}
    )
    summary_maxmse_df.to_csv(os.path.join(out_data_dir, "summary_maxmse.txt"))
    plot_max_mse_boxplot(maxmse_df, saveplot=True)

    runtime_df.to_csv(os.path.join(out_data_dir, "runtime.csv"), index=False)
    means = runtime_df.mean(axis=0)
    n = runtime_df.shape[0]
    stderr = runtime_df.std(axis=0, ddof=1) / np.sqrt(n)
    ci_lower = means - 1.96 * stderr
    ci_upper = means + 1.96 * stderr
    summary_runtime_df = pd.DataFrame(
        {"Runtime": means, "CI Lower": ci_lower, "CI Upper": ci_upper}
    )
    summary_runtime_df.to_csv(
        os.path.join(out_data_dir, "summary_runtime.txt")
    )

    # Second simulation
    # min_samples_leaf = [25, 50, 75, 100, 150, 200, 250, 300]
    min_samples_leaf = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    maxmse_msl_rf = np.zeros((nsim, len(min_samples_leaf)))
    maxmse_msl_magging = np.zeros((nsim, len(min_samples_leaf)))
    maxmse_msl_minmax = np.zeros((nsim, len(min_samples_leaf)))

    for j, msl in enumerate(tqdm(min_samples_leaf)):
        for i in range(nsim):
            dtr = gen_data_v6(n=n, noise_std=noise_std, random_state=i)
            Xtr = np.array(dtr.drop(columns=["E", "Y"]))
            Ytr = np.array(dtr["Y"])
            Etr = np.array(dtr["E"])

            # Default RF
            rf = RandomForest(
                "Regression",
                n_estimators=n_estimators,
                min_samples_leaf=msl,
                seed=i,
            )
            rf.fit(Xtr, Ytr)
            fitted_rf = rf.predict(Xtr)
            _, maxmse_rf = max_mse(Ytr, fitted_rf, Etr, ret_ind=True)
            maxmse_msl_rf[i, j] = maxmse_rf

            # MaggingRF
            rf_magging = MaggingRF_PB(
                n_estimators=n_estimators,
                min_samples_leaf=msl,
                random_state=i,
                backend="adaXT",
            )
            fitted_magging, _ = rf_magging.fit_predict_magging(
                Xtr, Ytr, Etr, Xtr
            )
            _, maxmse_magging = max_mse(Ytr, fitted_magging, Etr, ret_ind=True)
            maxmse_msl_magging[i, j] = maxmse_magging

            # MinMaxRF-M1 / Post-RF
            rf.modify_predictions_trees(Etr)
            fitted_minmax_m1 = rf.predict(Xtr)
            _, maxmse_minmax_m1 = max_mse(
                Ytr, fitted_minmax_m1, Etr, ret_ind=True
            )
            maxmse_msl_minmax[i, j] = maxmse_minmax_m1

    df_rf = pd.DataFrame(maxmse_msl_rf, columns=min_samples_leaf)
    df_rf = df_rf.melt(var_name="min_samples_leaf", value_name="MSE")
    df_rf["Method"] = "RF"

    df_magging = pd.DataFrame(maxmse_msl_magging, columns=min_samples_leaf)
    df_magging = df_magging.melt(var_name="min_samples_leaf", value_name="MSE")
    df_magging["Method"] = "MaggingRF"

    df_minmax = pd.DataFrame(maxmse_msl_minmax, columns=min_samples_leaf)
    df_minmax = df_minmax.melt(var_name="min_samples_leaf", value_name="MSE")
    df_minmax["Method"] = "Post-RF"

    stacked_df = pd.concat([df_rf, df_magging, df_minmax], ignore_index=True)

    stacked_df.to_csv(
        os.path.join(out_data_dir, "max_mse_msl.csv"), index=False
    )

    plot_max_mse_msl(stacked_df, saveplot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsim",
        type=int,
        default=20,
        help="Number of simulations to run (default: 20)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of observations (default: 1000)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.5,
        help="Standard deviation of the noise (default: 0.5)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=50,
        help="Number of trees in the Random Forest (default: 50)",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=30,
        help="The minimum number of observations required to be at a leaf node. (default: 30)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Name of the folder to save results (default: 'results')",
    )
    args = parser.parse_args()

    main(
        args.nsim,
        args.n,
        args.noise_std,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
    )
