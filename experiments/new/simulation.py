import argparse
import os
import copy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from nldg.new.utils import gen_data_v5, max_mse, min_xplvar
from nldg.new.rf import MaggingRF_PB
from adaXT.random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


def main(
    nsim: int,
    n_samples: int,
    adv_fraction: float,
    noise_var_env2: float,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
):
    results_dict = {
        "RF": [],
        "MaximinRF-Local": [],
        "MaximinRF-Global": [],
        "MaggingRF": [],
    }

    mse = copy.deepcopy(results_dict)
    mse_envs = copy.deepcopy(results_dict)
    xplvar_envs = copy.deepcopy(results_dict)
    maxmse = copy.deepcopy(results_dict)
    minxplvar = copy.deepcopy(results_dict)
    runtime = copy.deepcopy(results_dict)
    weights_magging = np.zeros((nsim, 2))

    for i in tqdm(range(nsim)):
        dtr = gen_data_v5(
            n_samples=n_samples,
            adv_fraction=adv_fraction,
            noise_var_env2=noise_var_env2,
            random_state=i,
        )
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        # Default RF
        start = time.time()
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=i,
        )
        rf.fit(Xtr, Ytr)
        end = time.time()
        runtime["RF"].append(end - start)
        fitted_rf = rf.predict(Xtr)
        mse["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mse_envs_rf, maxmse_rf = max_mse(Ytr, fitted_rf, Etr, ret_ind=True)
        mse_envs["RF"].append(mse_envs_rf)
        maxmse["RF"].append(maxmse_rf)
        xplvar_envs_rf, minxplvar_rf = min_xplvar(
            Ytr, fitted_rf, Etr, ret_ind=True
        )
        xplvar_envs["RF"].append(xplvar_envs_rf)
        minxplvar["RF"].append(minxplvar_rf)

        # Maximin RF - Local
        start = time.time()
        rf_adaxt = RandomForest(
            forest_type="MaximinLocal",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
        )
        rf_adaxt.fit(Xtr, Ytr, Etr)
        end = time.time()
        runtime["MaximinRF-Local"].append(end - start)
        fitted_adaxt = rf_adaxt.predict(Xtr)
        mse["MaximinRF-Local"].append(mean_squared_error(Ytr, fitted_adaxt))
        mse_envs_adaxt, maxmse_adaxt = max_mse(
            Ytr, fitted_adaxt, Etr, ret_ind=True
        )
        mse_envs["MaximinRF-Local"].append(mse_envs_adaxt)
        maxmse["MaximinRF-Local"].append(maxmse_adaxt)
        xplvar_envs_adaxt, minxplvar_adaxt = min_xplvar(
            Ytr, fitted_adaxt, Etr, ret_ind=True
        )
        xplvar_envs["MaximinRF-Local"].append(xplvar_envs_adaxt)
        minxplvar["MaximinRF-Local"].append(minxplvar_adaxt)

        # Maximin RF - Global
        start = time.time()
        rf_adaxt_global = RandomForest(
            forest_type="MaximinGlobal",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            seed=i,
        )
        rf_adaxt_global.fit(Xtr, Ytr, Etr)
        end = time.time()
        runtime["MaximinRF-Global"].append(end - start)
        fitted_adaxt_global = rf_adaxt_global.predict(Xtr)
        mse["MaximinRF-Global"].append(
            mean_squared_error(Ytr, fitted_adaxt_global)
        )
        mse_envs_adaxt_global, maxmse_adaxt_global = max_mse(
            Ytr, fitted_adaxt_global, Etr, ret_ind=True
        )
        mse_envs["MaximinRF-Global"].append(mse_envs_adaxt_global)
        maxmse["MaximinRF-Global"].append(maxmse_adaxt_global)
        xplvar_envs_adaxt_global, minxplvar_adaxt_global = min_xplvar(
            Ytr, fitted_adaxt_global, Etr, ret_ind=True
        )
        xplvar_envs["MaximinRF-Global"].append(xplvar_envs_adaxt_global)
        minxplvar["MaximinRF-Global"].append(minxplvar_adaxt_global)

        # Magging RF
        start = time.time()
        rf_magging = MaggingRF_PB(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            backend="sklearn",
            random_state=i,
        )
        fitted_magging, preds_magging = rf_magging.fit_predict_magging(
            Xtr, Ytr, Etr, Xtr
        )
        end = time.time()
        runtime["MaggingRF"].append(end - start)
        weights_magging[i, :] = rf_magging.get_weights()
        mse["MaggingRF"].append(mean_squared_error(Ytr, fitted_magging))
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
    mse_envs_df = get_df(mse_envs)
    xplvar_envs_df = get_df(xplvar_envs, mse=False)
    maxmse_df = pd.DataFrame(maxmse)
    minxplvar_df = pd.DataFrame(minxplvar)
    runtime_df = pd.DataFrame(runtime)
    weights_magging = pd.DataFrame(weights_magging)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    mse_df.to_csv(os.path.join(results_dir, "mse.csv"), index=False)
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
        "--n_samples",
        type=int,
        default=1000,
        help="Number of observations in the data (default: 1000)",
    )
    parser.add_argument(
        "--adv_fraction",
        type=float,
        default=0.1,
        help="Fraction of observations in the second environment (default: 0.1)",
    )
    parser.add_argument(
        "--noise_var_env2",
        type=float,
        default=10.0,
        help="Variance of the observations in the second environment (default: 10.0)",
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
        args.n_samples,
        args.adv_fraction,
        args.noise_var_env2,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
    )
