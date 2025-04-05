import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from nldg.new.archive.utils import gen_data_v3, min_xplvar
from nldg.new.archive.rf import MaggingRF, RF4DG, MaggingRF_PB
from adaXT.random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from experiments.new.archive.utils import (
    plot_mse_r2,
    plot_minxplvar,
    plot_weights_magging,
)


def main(
    nsim: int,
    n_train: int,
    n_test: int,
    train_setting: int,
    test_setting: int,
    method: int,
    n_estimators: int,
    min_samples_leaf: int,
    results_folder: str,
):
    mse_in = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF-Forest": [],
        "MaggingRF-Trees": [],
    }
    r2_in = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF-Forest": [],
        "MaggingRF-Trees": [],
    }
    mse_out = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF-Forest": [],
        "MaggingRF-Trees": [],
    }
    r2_out = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF-Forest": [],
        "MaggingRF-Trees": [],
    }
    minxplvar = {
        "RF": [],
        "MaximinRF": [],
        "MaggingRF-Forest": [],
        "MaggingRF-Trees": [],
    }
    # TODO: Maybe in the future we could generalize the code to arbitrary
    #  datasets. At the moment, it only considers 3 environments.
    weights_magging = np.zeros((nsim, 3))

    for i in tqdm(range(nsim)):
        dtr, dts = gen_data_v3(
            n_train=n_train,
            n_test=n_test,
            random_state=i,
            train_setting=train_setting,
            test_setting=test_setting,
        )
        Xtr, Xts = (
            np.array(dtr.drop(columns=["E", "Y"])),
            np.array(dts.drop(columns=["E", "Y"])),
        )
        Ytr, Yts = np.array(dtr["Y"]), np.array(dts["Y"])
        Etr = np.array(dtr["E"])

        # Default RF
        if method == 1:
            rf = RF4DG(
                criterion="mse",
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                disable=True,
                parallel=True,
                random_state=i,
            )
            rf.fit(Xtr, Ytr, Etr)
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=i,
            )
            # rf = RandomForest(
            #     forest_type="Regression",
            #     n_estimators=n_estimators,
            #     min_samples_leaf=min_samples_leaf,
            #     seed=i,
            # )
            rf.fit(Xtr, Ytr)
        preds_rf = rf.predict(Xts)
        fitted_rf = rf.predict(Xtr)

        # Maximin RF
        if method == 1:
            maximin_rf = RF4DG(
                criterion="maximin",
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                disable=True,
                parallel=True,
                random_state=i,
            )
            maximin_rf.fit(Xtr, Ytr, Etr)
        else:
            maximin_rf = RandomForest(
                forest_type="MaximinRegression",
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                seed=i,
            )
            maximin_rf.fit(Xtr, Ytr, Etr)
        preds_maximin_rf = maximin_rf.predict(Xts)
        fitted_maximin_rf = maximin_rf.predict(Xtr)

        # Magging RF
        if method == 1:
            magging_rf = MaggingRF_PB(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                disable=True,
                parallel=True,
                random_state=i,
            )
        else:
            magging_rf = MaggingRF_PB(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                backend="sklearn",
                random_state=i,
            )
            # magging_rf = MaggingRF_PB(
            #     n_estimators=n_estimators,
            #     min_samples_leaf=min_samples_leaf,
            #     backend="adaXT",
            #     random_state=i,
            # )
        fitted_magging_rf, preds_magging_rf = magging_rf.fit_predict_magging(
            Xtr, Ytr, Etr, Xts
        )
        weights_magging[i, :] = magging_rf.get_weights()

        # Magging RF 2
        magging_rf_2 = MaggingRF(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=i,
        )
        magging_rf_2.fit(Xtr, Ytr)
        preds_magging_rf_2, _ = magging_rf_2.predict_maximin(Xtr, Xts)
        fitted_magging_rf_2, _ = magging_rf_2.predict_maximin(Xtr, Xtr)

        # Save results
        mse_in["RF"].append(mean_squared_error(Ytr, fitted_rf))
        mse_in["MaximinRF"].append(mean_squared_error(Ytr, fitted_maximin_rf))
        mse_in["MaggingRF-Forest"].append(
            mean_squared_error(Ytr, fitted_magging_rf)
        )
        mse_in["MaggingRF-Trees"].append(
            mean_squared_error(Ytr, fitted_magging_rf_2)
        )

        r2_in["RF"].append(r2_score(Ytr, fitted_rf))
        r2_in["MaximinRF"].append(r2_score(Ytr, fitted_maximin_rf))
        r2_in["MaggingRF-Forest"].append(r2_score(Ytr, fitted_magging_rf))
        r2_in["MaggingRF-Trees"].append(r2_score(Ytr, fitted_magging_rf_2))

        mse_out["RF"].append(mean_squared_error(Yts, preds_rf))
        mse_out["MaximinRF"].append(mean_squared_error(Yts, preds_maximin_rf))
        mse_out["MaggingRF-Forest"].append(
            mean_squared_error(Yts, preds_magging_rf)
        )
        mse_out["MaggingRF-Trees"].append(
            mean_squared_error(Yts, preds_magging_rf_2)
        )

        r2_out["RF"].append(r2_score(Yts, preds_rf))
        r2_out["MaximinRF"].append(r2_score(Yts, preds_maximin_rf))
        r2_out["MaggingRF-Forest"].append(r2_score(Yts, preds_magging_rf))
        r2_out["MaggingRF-Trees"].append(r2_score(Yts, preds_magging_rf_2))

        minxplvar["RF"].append(min_xplvar(Ytr, fitted_rf, Etr))
        minxplvar["MaximinRF"].append(min_xplvar(Ytr, fitted_maximin_rf, Etr))
        minxplvar["MaggingRF-Forest"].append(
            min_xplvar(Ytr, fitted_magging_rf, Etr)
        )
        minxplvar["MaggingRF-Trees"].append(
            min_xplvar(Ytr, fitted_magging_rf_2, Etr)
        )

    # Plot and save
    mse_in_df = pd.DataFrame(mse_in)
    r2_in_df = pd.DataFrame(r2_in)
    mse_out_df = pd.DataFrame(mse_out)
    r2_out_df = pd.DataFrame(r2_out)
    minxplvar_df = pd.DataFrame(minxplvar)
    weights_magging = pd.DataFrame(weights_magging)

    results_dir = os.path.join(os.path.dirname(__file__), results_folder)
    os.makedirs(results_dir, exist_ok=True)

    plot_mse_r2(
        mse_in_df,
        r2_in_df,
        "sim_mse_r2_in.pdf",
        results_dir,
        out=False,
    )
    plot_mse_r2(mse_out_df, r2_out_df, "sim_mse_r2_out.pdf", results_dir)
    plot_minxplvar(minxplvar_df, "sim_minxplvar.pdf", results_dir)
    plot_weights_magging(
        weights_magging, "sim_weights_magging.pdf", results_dir
    )

    mse_in_df.to_csv(os.path.join(results_dir, "mse_in.csv"), index=False)
    r2_in_df.to_csv(os.path.join(results_dir, "r2_in.csv"), index=False)
    mse_out_df.to_csv(os.path.join(results_dir, "mse_out.csv"), index=False)
    r2_out_df.to_csv(os.path.join(results_dir, "r2_out.csv"), index=False)
    minxplvar_df.to_csv(
        os.path.join(results_dir, "min_xpl_var.csv"), index=False
    )
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
        "--train_setting",
        type=int,
        default=1,
        help="Train data setting. Value in {1,2} (default: 1)",
    )
    parser.add_argument(
        "--test_setting",
        type=int,
        default=1,
        help="Test data setting. Value in {1,2} (default: 1)",
    )
    parser.add_argument(
        "--method",
        type=int,
        default=1,
        help="1 for the inefficient version, 2 for the efficient one (default: 1)",
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
        args.train_setting,
        args.test_setting,
        args.method,
        args.n_estimators,
        args.min_samples_leaf,
        args.results_folder,
    )
