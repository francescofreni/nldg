import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from nldg.new.utils import gen_data_v3, max_mse
from nldg.new.train_nn import train_model, train_model_GDRO, predict_GDRO
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
    train_setting: int,
    test_setting: int,
    epochs: int,
    results_folder: str,
):
    mse_in = {
        "NN": [],
        "MaximinNN": [],
        "MaggingNN": [],
        "GroupDRO": [],
    }
    r2_in = {
        "NN": [],
        "MaximinNN": [],
        "MaggingNN": [],
        "GroupDRO": [],
    }
    mse_out = {
        "NN": [],
        "MaximinNN": [],
        "MaggingNN": [],
        "GroupDRO": [],
    }
    r2_out = {
        "NN": [],
        "MaximinNN": [],
        "MaggingNN": [],
        "GroupDRO": [],
    }
    maxmse = {
        "NN": [],
        "MaximinNN": [],
        "MaggingNN": [],
        "GroupDRO": [],
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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(Xtr)
        X_test = scaler.transform(Xts)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # Default NN
        model = train_model(
            X_train, Ytr, Etr, epochs=epochs, verbose=False, default=True
        )
        model.eval()
        with torch.no_grad():
            preds_nn = model(X_test_tensor).numpy()
            fitted_nn = model(X_train_tensor).numpy()

        # Maximin NN
        model = train_model(X_train, Ytr, Etr, epochs=epochs, verbose=False)
        model.eval()
        with torch.no_grad():
            preds_maximin_nn = model(X_test_tensor).numpy()
            fitted_maximin_nn = model(X_train_tensor).numpy()

        # Magging NN
        n_envs = len(np.unique(Etr))
        winit = np.array([1 / n_envs] * n_envs)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(n_envs)]

        preds_envs = []
        fitted_envs = []
        for env in np.unique(Etr):
            Xtr_e = X_train[Etr == env]
            Ytr_e = Ytr[Etr == env]
            model = train_model(
                Xtr_e,
                Ytr_e,
                Etr[Etr == env],
                epochs=epochs,
                verbose=False,
                default=True,
            )
            model.eval()
            with torch.no_grad():
                preds_envs.append(model(X_test_tensor).numpy())
                fitted_envs.append(model(X_train_tensor).numpy())
        preds_envs = np.column_stack(preds_envs)
        fitted_envs = np.column_stack(fitted_envs)

        wmag = minimize(
            objective,
            winit,
            args=(fitted_envs,),
            bounds=bounds,
            constraints=constraints,
        ).x
        preds_magging_nn = np.dot(wmag, preds_envs.T)
        fitted_magging_nn = np.dot(wmag, fitted_envs.T)
        weights_magging[i, :] = wmag

        # Group DRO
        model, bweights = train_model_GDRO(X_train, Ytr, Etr, lr_model=0.01)
        preds_gdro = predict_GDRO(model, X_test)
        fitted_gdro = predict_GDRO(model, X_train)

        # Save results
        mse_in["NN"].append(mean_squared_error(Ytr, fitted_nn))
        mse_in["MaximinNN"].append(mean_squared_error(Ytr, fitted_maximin_nn))
        mse_in["MaggingNN"].append(mean_squared_error(Ytr, fitted_magging_nn))
        mse_in["GroupDRO"].append(mean_squared_error(Ytr, fitted_gdro))

        r2_in["NN"].append(r2_score(Ytr, fitted_nn))
        r2_in["MaximinNN"].append(r2_score(Ytr, fitted_maximin_nn))
        r2_in["MaggingNN"].append(r2_score(Ytr, fitted_magging_nn))
        r2_in["GroupDRO"].append(r2_score(Ytr, fitted_gdro))

        mse_out["NN"].append(mean_squared_error(Yts, preds_nn))
        mse_out["MaximinNN"].append(mean_squared_error(Yts, preds_maximin_nn))
        mse_out["MaggingNN"].append(mean_squared_error(Yts, preds_magging_nn))
        mse_out["GroupDRO"].append(mean_squared_error(Yts, preds_gdro))

        r2_out["NN"].append(r2_score(Yts, preds_nn))
        r2_out["MaximinNN"].append(r2_score(Yts, preds_maximin_nn))
        r2_out["MaggingNN"].append(r2_score(Yts, preds_magging_nn))
        r2_out["GroupDRO"].append(r2_score(Yts, preds_gdro))

        maxmse["NN"].append(max_mse(Ytr, fitted_nn, Etr))
        maxmse["MaximinNN"].append(max_mse(Ytr, fitted_maximin_nn, Etr))
        maxmse["MaggingNN"].append(max_mse(Ytr, fitted_magging_nn, Etr))
        maxmse["GroupDRO"].append(max_mse(Ytr, fitted_gdro, Etr))

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
        nn=True,
    )
    plot_mse_r2(
        mse_out_df, r2_out_df, "sim_mse_r2_out.pdf", results_dir, nn=True
    )
    plot_maxmse(maxmse_df, "sim_maxmse.pdf", results_dir, nn=True)
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
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs. (default: 100)",
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
        args.epochs,
        args.results_folder,
    )
