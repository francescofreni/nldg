import argparse
import os
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import r2_score, mean_squared_error
from nldg.new.utils import gen_data_isd_v3
from nldg.new.rf import RF4DL, MaggingRF, IsdRF
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from tqdm import tqdm
from experiments.new.utils import plot_mse_r2_adapt


def objective(w: np.ndarray, F: np.ndarray) -> float:
    return np.dot(w.T, np.dot(F.T, F).dot(w))


def main(
    nsim: int,
    n_train: int,
    n_test: int,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
    prop_ad: float,
):
    results_dict = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF": [],
        "MaggingRF2": [],
        "IsdRF": [],
        "IsdRFad": [],
        "RFad": [],
    }
    mse_in = copy.deepcopy(results_dict)
    mse_out = copy.deepcopy(results_dict)
    r2_in = copy.deepcopy(results_dict)
    r2_out = copy.deepcopy(results_dict)

    for i in tqdm(range(nsim)):
        dtr, dad, dts = gen_data_isd_v3(
            n_train=n_train,
            n_test=n_test,
            random_state=i,
            prop_ad=prop_ad,
        )
        Xtr, Xad, Xts = (
            np.array(dtr.drop(columns=["E", "Y"])),
            np.array(dad.drop(columns=["E", "Y"])),
            np.array(dts.drop(columns=["E", "Y"])),
        )
        Ytr, Yad, Yts = (
            np.array(dtr["Y"]),
            np.array(dad["Y"]),
            np.array(dts["Y"]),
        )
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
        isd_rf = IsdRF(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf
        )
        isd_rf.find_invariant(Xtr, Ytr, Etr)
        preds_isd = isd_rf.predict_zeroshot(Xts)
        fitted_isd = isd_rf.predict_zeroshot(Xtr)

        # ISD RF - adaptation step
        preds_isd_ad = isd_rf.adaption(Xad, Yad, Xts)

        # Default RF - adaptation step
        rf_ad = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        rf_ad.fit(Xad, Yad)
        preds_rf_ad = rf_ad.predict(Xts)

        # Save results
        mse_in["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mse_in["MaximinRF"].append(mean_squared_error(Ytr, fitted_maximin_rf))
        mse_in["MaggingRF"].append(mean_squared_error(Ytr, fitted_magging_rf))
        mse_in["MaggingRF2"].append(
            mean_squared_error(Ytr, fitted_magging_rf_2)
        )
        mse_in["IsdRF"].append(mean_squared_error(Ytr, fitted_isd))
        mse_in["IsdRFad"].append(mean_squared_error(Ytr, fitted_isd))
        mse_in["RFad"].append(mean_squared_error(Ytr, fitted_rf))

        r2_in["RF"].append(r2_score(Ytr, fitted_rf))
        r2_in["MaximinRF"].append(r2_score(Ytr, fitted_maximin_rf))
        r2_in["MaggingRF"].append(r2_score(Ytr, fitted_magging_rf))
        r2_in["MaggingRF2"].append(r2_score(Ytr, fitted_magging_rf_2))
        r2_in["IsdRF"].append(r2_score(Ytr, fitted_isd))
        r2_in["IsdRFad"].append(r2_score(Ytr, fitted_isd))
        r2_in["RFad"].append(r2_score(Ytr, fitted_rf))

        mse_out["RF"].append(mean_squared_error(Yts, preds_rf))
        mse_out["MaximinRF"].append(mean_squared_error(Yts, preds_maximin_rf))
        mse_out["MaggingRF"].append(mean_squared_error(Yts, preds_magging_rf))
        mse_out["MaggingRF2"].append(
            mean_squared_error(Yts, preds_magging_rf_2)
        )
        mse_out["IsdRF"].append(mean_squared_error(Yts, preds_isd))
        mse_out["IsdRFad"].append(mean_squared_error(Yts, preds_isd_ad))
        mse_out["RFad"].append(mean_squared_error(Yts, preds_rf_ad))

        r2_out["RF"].append(r2_score(Yts, preds_rf))
        r2_out["MaximinRF"].append(r2_score(Yts, preds_maximin_rf))
        r2_out["MaggingRF"].append(r2_score(Yts, preds_magging_rf))
        r2_out["MaggingRF2"].append(r2_score(Yts, preds_magging_rf_2))
        r2_out["IsdRF"].append(r2_score(Yts, preds_isd))
        r2_out["IsdRFad"].append(r2_score(Yts, preds_isd_ad))
        r2_out["RFad"].append(r2_score(Yts, preds_rf_ad))

    # Plot and save
    mse_in_df = pd.DataFrame(mse_in)
    r2_in_df = pd.DataFrame(r2_in)
    mse_out_df = pd.DataFrame(mse_out)
    r2_out_df = pd.DataFrame(r2_out)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    plot_mse_r2_adapt(
        mse_in_df,
        r2_in_df,
        "sim_mse_r2_in.pdf",
        results_dir,
        out=False,
    )
    plot_mse_r2_adapt(
        mse_out_df,
        r2_out_df,
        "sim_mse_r2_out.pdf",
        results_dir,
    )

    mse_in_df.to_csv(os.path.join(results_dir, "mse_in.csv"), index=False)
    r2_in_df.to_csv(os.path.join(results_dir, "r2_in.csv"), index=False)
    mse_out_df.to_csv(os.path.join(results_dir, "mse_out.csv"), index=False)
    r2_out_df.to_csv(os.path.join(results_dir, "r2_out.csv"), index=False)


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
    parser.add_argument(
        "--prop_ad",
        type=float,
        default=0.1,
        help="Proportion of test data destined to adaptation step (default: 0.1)",
    )
    args = parser.parse_args()

    main(
        args.nsim,
        args.n_train,
        args.n_test,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
        args.prop_ad,
    )
