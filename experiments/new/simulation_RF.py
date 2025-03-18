import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from nldg.new.utils import gen_data_v3, max_mse, gen_data_isd, gen_data_isd_v2
from nldg.new.rf import RF4DL, MaggingRF, IsdRF
from scipy.optimize import minimize
from tqdm import tqdm
from experiments.new.utils import (
    plot_mse_r2,
    plot_maxmse,
    plot_weights_magging,
)


def objective(w: np.ndarray, F: np.ndarray) -> float:
    return np.dot(w.T, np.dot(F.T, F).dot(w))


def main(
    nsim: int,
    n_train: int,
    n_test: int,
    setting: int,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
    isd: bool,
    isd_genfun: int,
):
    if isd:
        mse_in = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
            "IsdRF": [],
        }
        r2_in = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
            "IsdRF": [],
        }
        mse_out = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
            "IsdRF": [],
        }
        r2_out = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
            "IsdRF": [],
        }
        maxmse = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
            "IsdRF": [],
        }
    else:
        mse_in = {"RF": [], "MaximinRF": [], "MaggingRF": [], "MaggingRF2": []}
        r2_in = {"RF": [], "MaximinRF": [], "MaggingRF": [], "MaggingRF2": []}
        mse_out = {
            "RF": [],
            "MaximinRF": [],
            "MaggingRF": [],
            "MaggingRF2": [],
        }
        r2_out = {"RF": [], "MaximinRF": [], "MaggingRF": [], "MaggingRF2": []}
        maxmse = {"RF": [], "MaximinRF": [], "MaggingRF": [], "MaggingRF2": []}
    # TODO: Maybe in the future we could generalize the code to arbitrary
    #  datasets. At the moment, it only considers 3 environments.
    weights_magging = np.zeros((nsim, 3))

    for i in tqdm(range(nsim)):
        if isd:
            if isd_genfun == 1:
                dtr, dts = gen_data_isd(
                    n_train=n_train,
                    n_test=n_test,
                    random_state=i,
                )
            else:
                dtr, dts = gen_data_isd_v2(
                    n_train=n_train,
                    n_test=n_test,
                    random_state=i,
                )
        else:
            dtr, dts = gen_data_v3(
                n_train=n_train, n_test=n_test, random_state=i, setting=setting
            )
        Xtr, Xts = (
            np.array(dtr.drop(columns=["E", "Y"])),
            np.array(dts.drop(columns=["E", "Y"])),
        )
        Ytr, Yts = np.array(dtr["Y"]), np.array(dts["Y"])
        Etr = np.array(dtr["E"])

        # Default RF
        rf = RF4DL(
            criterion="mse",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            disable=True,
        )
        rf.fit(Xtr, Ytr, Etr)
        preds_rf = rf.predict(Xts)
        fitted_rf = rf.predict(Xtr)

        # Maximin RF
        maximin_rf = RF4DL(
            criterion="maximin",
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            disable=True,
        )
        maximin_rf.fit(Xtr, Ytr, Etr)
        preds_maximin_rf = maximin_rf.predict(Xts)
        fitted_maximin_rf = maximin_rf.predict(Xtr)

        # Magging RF
        n_envs = len(np.unique(Etr))
        winit = np.array([1 / n_envs] * n_envs)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(n_envs)]
        preds_envs = []
        fitted_envs = []
        for env in np.unique(Etr):
            Xtr_e = Xtr[Etr == env]
            Ytr_e = Ytr[Etr == env]
            Etr_e = Etr[Etr == env]
            magging_rf = RF4DL(
                criterion="mse",
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                disable=True,
            )
            magging_rf.fit(Xtr_e, Ytr_e, Etr_e)
            preds_envs.append(magging_rf.predict(Xts))
            fitted_envs.append(magging_rf.predict(Xtr))
        preds_envs = np.column_stack(preds_envs)
        fitted_envs = np.column_stack(fitted_envs)
        wmag = minimize(
            objective,
            winit,
            args=(fitted_envs,),
            bounds=bounds,
            constraints=constraints,
        ).x
        weights_magging[i, :] = wmag
        preds_magging_rf = np.dot(wmag, preds_envs.T)
        fitted_magging_rf = np.dot(wmag, fitted_envs.T)

        # Magging RF 2
        magging_rf_2 = MaggingRF(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf
        )
        magging_rf_2.fit(Xtr, Ytr)
        preds_magging_rf_2 = magging_rf_2.predict(Xts)
        fitted_magging_rf_2 = magging_rf_2.predict(Xtr)

        # ISD RF
        if isd:
            isd_rf = IsdRF(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf
            )
            isd_rf.find_invariant(Xtr, Ytr, Etr)
            preds_isd = isd_rf.predict_zeroshot(Xts)
            fitted_isd = isd_rf.predict_zeroshot(Xtr)

        # Save results
        mse_in["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mse_in["MaximinRF"].append(mean_squared_error(Ytr, fitted_maximin_rf))
        mse_in["MaggingRF"].append(mean_squared_error(Ytr, fitted_magging_rf))
        mse_in["MaggingRF2"].append(
            mean_squared_error(Ytr, fitted_magging_rf_2)
        )

        r2_in["RF"].append(r2_score(Ytr, fitted_rf))
        r2_in["MaximinRF"].append(r2_score(Ytr, fitted_maximin_rf))
        r2_in["MaggingRF"].append(r2_score(Ytr, fitted_magging_rf))
        r2_in["MaggingRF2"].append(r2_score(Ytr, fitted_magging_rf_2))

        mse_out["RF"].append(mean_squared_error(Yts, preds_rf))
        mse_out["MaximinRF"].append(mean_squared_error(Yts, preds_maximin_rf))
        mse_out["MaggingRF"].append(mean_squared_error(Yts, preds_magging_rf))
        mse_out["MaggingRF2"].append(
            mean_squared_error(Yts, preds_magging_rf_2)
        )

        r2_out["RF"].append(r2_score(Yts, preds_rf))
        r2_out["MaximinRF"].append(r2_score(Yts, preds_maximin_rf))
        r2_out["MaggingRF"].append(r2_score(Yts, preds_magging_rf))
        r2_out["MaggingRF2"].append(r2_score(Yts, preds_magging_rf_2))

        maxmse["RF"].append(max_mse(Ytr, fitted_rf, Etr))
        maxmse["MaximinRF"].append(max_mse(Ytr, fitted_maximin_rf, Etr))
        maxmse["MaggingRF"].append(max_mse(Ytr, fitted_magging_rf, Etr))
        maxmse["MaggingRF2"].append(max_mse(Ytr, fitted_magging_rf_2, Etr))

        if isd:
            mse_in["IsdRF"].append(mean_squared_error(Ytr, fitted_isd))
            r2_in["IsdRF"].append(r2_score(Ytr, fitted_isd))
            mse_out["IsdRF"].append(mean_squared_error(Yts, preds_isd))
            r2_out["IsdRF"].append(r2_score(Yts, preds_isd))
            maxmse["IsdRF"].append(max_mse(Ytr, fitted_isd, Etr))

    # Plot and save
    mse_in_df = pd.DataFrame(mse_in)
    r2_in_df = pd.DataFrame(r2_in)
    mse_out_df = pd.DataFrame(mse_out)
    r2_out_df = pd.DataFrame(r2_out)
    maxmse_df = pd.DataFrame(maxmse)
    weights_magging = pd.DataFrame(weights_magging)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    plot_mse_r2(
        mse_in_df,
        r2_in_df,
        "sim_mse_r2_in.pdf",
        results_dir,
        out=False,
        isd=isd,
    )
    plot_mse_r2(
        mse_out_df, r2_out_df, "sim_mse_r2_out.pdf", results_dir, isd=isd
    )
    plot_maxmse(maxmse_df, "sim_maxmse.pdf", results_dir, isd=isd)
    plot_weights_magging(
        weights_magging, "sim_weights_magging.pdf", results_dir
    )

    mse_in_df.to_csv(os.path.join(results_dir, "mse_in.csv"), index=False)
    r2_in_df.to_csv(os.path.join(results_dir, "r2_in.csv"), index=False)
    mse_out_df.to_csv(os.path.join(results_dir, "mse_out.csv"), index=False)
    r2_out_df.to_csv(os.path.join(results_dir, "r2_out.csv"), index=False)
    maxmse_df.to_csv(os.path.join(results_dir, "maxmse.csv"), index=False)
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
        "--setting",
        type=int,
        default=1,
        help="Data setting. Value in {1,2} (default: 1)",
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
    parser.add_argument(
        "--isd",
        type=bool,
        default=False,
        help="Whether to include ISD (default: False)",
    )
    parser.add_argument(
        "--isd_genfun",
        type=int,
        default=1,
        help="If isd, function to generate data. Value in {1,2} (default: 1)",
    )
    args = parser.parse_args()

    main(
        args.nsim,
        args.n_train,
        args.n_test,
        args.setting,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
        args.isd,
        args.isd_genfun,
    )
