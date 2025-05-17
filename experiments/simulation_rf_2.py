import argparse
import os
from nldg.utils import *
from nldg.rf import MaggingRF_PB
from adaXT.random_forest import RandomForest
from tqdm import tqdm


def main(
    nsim: int,
    n: int,
    noise_std: float,
    n_estimators: int,
    results_folder: str,
):
    min_samples_leaf = [25, 50, 75, 100, 150, 200, 250, 300]
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

            # MinMaxRF-M1
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
    df_magging["Method"] = "Magging"

    df_minmax = pd.DataFrame(maxmse_msl_minmax, columns=min_samples_leaf)
    df_minmax = df_minmax.melt(var_name="min_samples_leaf", value_name="MSE")
    df_minmax["Method"] = "MinMax"

    stacked_df = pd.concat([df_rf, df_magging, df_minmax], ignore_index=True)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    stacked_df.to_csv(
        os.path.join(results_dir, "max_mse_msl.csv"), index=False
    )


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
        args.results_folder,
    )
