import argparse
import os
import copy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from nldg.new.utils import gen_data, max_mse, min_xplvar
from nldg.new.rf import RF4DG, MaggingRF_PB
from adaXT.random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


def main(
    nsim: int,
    n_train: int,
    n_test: int,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
):
    results_dict = {
        "RF": [],
        "MaximinRF": [],
        "MaximinRF-adaXT": [],
        "MaximinRF-adaXT-Global": [],
        "MaggingRF": [],
    }

    mse = copy.deepcopy(results_dict)
    mspe_1 = copy.deepcopy(results_dict)
    mspe_2 = copy.deepcopy(results_dict)
    mse_envs = copy.deepcopy(results_dict)
    xplvar_envs = copy.deepcopy(results_dict)
    maxmse = copy.deepcopy(results_dict)
    minxplvar = copy.deepcopy(results_dict)
    runtime = copy.deepcopy(results_dict)
    weights_magging = np.zeros((nsim, 3))

    for i in tqdm(range(nsim)):
        dtr, dts1, dts2 = gen_data(
            n_train=n_train,
            n_test=n_test,
            random_state=i,
        )
        Xtr, Xts1, Xts2 = (
            np.array(dtr.drop(columns=["E", "Y"])),
            np.array(dts1.drop(columns=["E", "Y"])),
            np.array(dts2.drop(columns=["E", "Y"])),
        )  # Xts1 and Xts2 are the same
        Ytr, Yts1, Yts2 = (
            np.array(dtr["Y"]),
            np.array(dts1["Y"]),
            np.array(dts2["Y"]),
        )
        Etr = np.array(dtr["E"])

        # Default RF
        start = time.time()
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=i,
        )
        end = time.time()
        runtime["RF"].append(end - start)
        rf.fit(Xtr, Ytr)
        preds_rf = rf.predict(Xts1)
        fitted_rf = rf.predict(Xtr)
        mse["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mspe_1["RF"].append(mean_squared_error(Yts1, preds_rf))
        mspe_2["RF"].append(mean_squared_error(Yts2, preds_rf))
        mse_envs_rf, maxmse_rf = max_mse(Ytr, fitted_rf, Etr, ret_ind=True)
        mse_envs["RF"].append(mse_envs_rf)
        maxmse["RF"].append(maxmse_rf)
        xplvar_envs_rf, minxplvar_rf = min_xplvar(
            Ytr, fitted_rf, Etr, ret_ind=True
        )
        xplvar_envs["RF"].append(xplvar_envs_rf)
        minxplvar["RF"].append(minxplvar_rf)

        # Maximin RF
        start = time.time()
        rf_maximin = RF4DG(
            criterion="maximin",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            disable=True,
            parallel=True,
            random_state=i,
        )
        end = time.time()
        runtime["MaximinRF"].append(end - start)
        rf_maximin.fit(Xtr, Ytr, Etr)
        preds_maximin = rf_maximin.predict(Xts1)
        fitted_maximin = rf_maximin.predict(Xtr)
        mse["MaximinRF"].append(mean_squared_error(Ytr, fitted_maximin))
        mspe_1["MaximinRF"].append(mean_squared_error(Yts1, preds_maximin))
        mspe_2["MaximinRF"].append(mean_squared_error(Yts2, preds_maximin))
        mse_envs_maximin, maxmse_maximin = max_mse(
            Ytr, fitted_maximin, Etr, ret_ind=True
        )
        mse_envs["MaximinRF"].append(mse_envs_maximin)
        maxmse["MaximinRF"].append(maxmse_maximin)
        xplvar_envs_maximin, minxplvar_maximin = min_xplvar(
            Ytr, fitted_maximin, Etr, ret_ind=True
        )
        xplvar_envs["MaximinRF"].append(xplvar_envs_maximin)
        minxplvar["MaximinRF"].append(minxplvar_maximin)

        # Maximin RF - adaXT
        start = time.time()
        rf_adaxt = RandomForest(
            forest_type="MaximinRegression",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
        )
        end = time.time()
        runtime["MaximinRF-adaXT"].append(end - start)
        rf_adaxt.fit(Xtr, Ytr, Etr)
        preds_adaxt = rf_adaxt.predict(Xts1)
        fitted_adaxt = rf_adaxt.predict(Xtr)
        mse["MaximinRF-adaXT"].append(mean_squared_error(Ytr, fitted_adaxt))
        mspe_1["MaximinRF-adaXT"].append(mean_squared_error(Yts1, preds_adaxt))
        mspe_2["MaximinRF-adaXT"].append(mean_squared_error(Yts2, preds_adaxt))
        mse_envs_adaxt, maxmse_adaxt = max_mse(
            Ytr, fitted_adaxt, Etr, ret_ind=True
        )
        mse_envs["MaximinRF-adaXT"].append(mse_envs_adaxt)
        maxmse["MaximinRF-adaXT"].append(maxmse_adaxt)
        xplvar_envs_adaxt, minxplvar_adaxt = min_xplvar(
            Ytr, fitted_adaxt, Etr, ret_ind=True
        )
        xplvar_envs["MaximinRF-adaXT"].append(xplvar_envs_adaxt)
        minxplvar["MaximinRF-adaXT"].append(minxplvar_adaxt)

        # Maximin RF - adaXT - Global
        start = time.time()
        rf_adaxt_global = RandomForest(
            forest_type="MaximinRegression_Global",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
        )
        end = time.time()
        runtime["MaximinRF-adaXT-Global"].append(end - start)
        rf_adaxt_global.fit(Xtr, Ytr, Etr)
        preds_adaxt_global = rf_adaxt_global.predict(Xts1)
        fitted_adaxt_global = rf_adaxt_global.predict(Xtr)
        mse["MaximinRF-adaXT-Global"].append(
            mean_squared_error(Ytr, fitted_adaxt_global)
        )
        mspe_1["MaximinRF-adaXT-Global"].append(
            mean_squared_error(Yts1, preds_adaxt_global)
        )
        mspe_2["MaximinRF-adaXT-Global"].append(
            mean_squared_error(Yts2, preds_adaxt_global)
        )
        mse_envs_adaxt_global, maxmse_adaxt_global = max_mse(
            Ytr, fitted_adaxt_global, Etr, ret_ind=True
        )
        mse_envs["MaximinRF-adaXT-Global"].append(mse_envs_adaxt_global)
        maxmse["MaximinRF-adaXT-Global"].append(maxmse_adaxt_global)
        xplvar_envs_adaxt_global, minxplvar_adaxt_global = min_xplvar(
            Ytr, fitted_adaxt_global, Etr, ret_ind=True
        )
        xplvar_envs["MaximinRF-adaXT-Global"].append(xplvar_envs_adaxt_global)
        minxplvar["MaximinRF-adaXT-Global"].append(minxplvar_adaxt_global)

        # Magging RF
        start = time.time()
        rf_magging = MaggingRF_PB(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            backend="sklearn",
            random_state=i,
        )
        fitted_magging, preds_magging = rf_magging.fit_predict_magging(
            Xtr, Ytr, Etr, Xts1
        )
        end = time.time()
        runtime["MaggingRF"].append(end - start)
        weights_magging[i, :] = rf_magging.get_weights()
        mse["MaggingRF"].append(mean_squared_error(Ytr, fitted_magging))
        mspe_1["MaggingRF"].append(mean_squared_error(Yts1, preds_magging))
        mspe_2["MaggingRF"].append(mean_squared_error(Yts2, preds_magging))
        mse_envs_magging, maxmse_magging = max_mse(
            Ytr, fitted_magging, Etr, ret_ind=True
        )
        mse_envs["MaggingRF"].append(mse_envs_magging)
        maxmse["MaggingRF"].append(maxmse_magging)
        xplvar_envs_magging, minxplvar_magging = min_xplvar(
            Ytr, fitted_magging, Etr, ret_ind=True
        )
        xplvar_envs["MaggingRF"].append(xplvar_envs_magging)
        minxplvar["MaggingRF"].append(minxplvar_magging)

    def get_df(res_dict, mse=True):
        rows = []
        for method, lists in res_dict.items():
            for sim_id, res_envs in enumerate(lists):
                for env_id, res in enumerate(res_envs):
                    if mse:
                        rows.append(
                            {
                                "MSE": res,
                                "env_id": env_id,
                                "sim_id": sim_id,
                                "method": method,
                            }
                        )
                    else:
                        rows.append(
                            {
                                "xplvar": res,
                                "env_id": env_id,
                                "sim_id": sim_id,
                                "method": method,
                            }
                        )
        return pd.DataFrame(rows)

    mse_df = pd.DataFrame(mse)
    mspe_1_df = pd.DataFrame(mspe_1)
    mspe_2_df = pd.DataFrame(mspe_2)
    mse_envs_df = get_df(mse_envs)
    xplvar_envs_df = get_df(xplvar_envs, mse=False)
    maxmse_df = pd.DataFrame(maxmse)
    minxplvar_df = pd.DataFrame(minxplvar)
    runtime_df = pd.DataFrame(runtime)
    weights_magging = pd.DataFrame(weights_magging)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    mse_df.to_csv(os.path.join(results_dir, "mse.csv"), index=False)
    mspe_1_df.to_csv(os.path.join(results_dir, "mspe_1.csv"), index=False)
    mspe_2_df.to_csv(os.path.join(results_dir, "mspe_2.csv"), index=False)
    mse_envs_df.to_csv(os.path.join(results_dir, "mse_envs.csv"), index=False)
    xplvar_envs_df.to_csv(
        os.path.join(results_dir, "xplvar_envs.csv"), index=False
    )
    maxmse_df.to_csv(os.path.join(results_dir, "max_mse.csv"), index=False)
    minxplvar_df.to_csv(
        os.path.join(results_dir, "min_xpl_var.csv"), index=False
    )
    runtime_df.to_csv(os.path.join(results_dir, "runtime.csv"), index=False)
    weights_magging.to_csv(
        os.path.join(results_dir, "weights_magging.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsim",
        type=int,
        default=100,
        help="Number of simulations to run (default: 100)",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=1000,
        help="Number of observations in the training data (default: 1000)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=500,
        help="Number of observations in the test data (default: 500)",
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
        default=10,
        help="The minimum number of observations required to be at a leaf node. (default: 10)",
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
        args.n_train,
        args.n_test,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
    )
