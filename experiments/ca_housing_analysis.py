import os
import copy
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from nldg.utils import max_mse, max_regret, min_reward
from adaXT.random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from utils import (
    plot_test_risk,
    plot_envs_risk,
    plot_test_risk_all_methods,
    table_test_risk_all_methods,
    plot_envs_mse_all_methods,
    write_lr_test_table_txt,
    write_lr_env_specific_table_txt,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_ca_housing")
os.makedirs(OUT_DIR, exist_ok=True)

# QUADRANTS = ["SW", "SE", "NW", "NE"]
QUADRANTS = ["Env 1", "Env 2", "Env 3", "Env 4"]
B = 20
VAL_PERCENTAGE = 0.3
N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
N = 20

NAME_RF = "MaxRM-RF"


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and preprocess the California housing dataset.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        Z (pd.DataFrame): Additional covariates
    """
    data_path = os.path.join(DATA_DIR, "housing-sklearn.csv")
    dat = pd.read_csv(filepath_or_buffer=data_path)

    y = dat["MedHouseVal"]
    Z = dat[["Latitude", "Longitude"]]
    X = dat.drop(["MedHouseVal", "Latitude", "Longitude"], axis=1)

    return X, y, Z


def assign_quadrant(
    Z: pd.DataFrame,
) -> np.ndarray:
    """
    Creates the environment label based on geographic criteria.

    Args:
        Z (pd.DataFrame): Additional covariates (Latitude, Longitude)

    Returns:
        env (np.ndarray): Environment label
    """
    lat, lon = Z["Latitude"], Z["Longitude"]

    # north = lat >= 35
    # south = ~north
    # east = lon >= -120
    # west = ~east
    #
    # env = np.zeros(len(Z), dtype=int)
    # env[south & west] = 0  # SW
    # env[south & east] = 1  # SE
    # env[north & west] = 2  # NW
    # env[north & east] = 3  # NE

    west = lon < -121.5
    east = ~west
    sw = (lat < 38) & west
    nw = (lat >= 38) & west
    lat_thr = 34.5
    se = (lat < lat_thr) & east
    ne = (lat >= lat_thr) & east

    env = np.zeros(len(Z), dtype=int)
    env[sw] = 0  # SW
    env[se] = 1  # SE
    env[nw] = 2  # NW
    env[ne] = 3  # NE

    return env


def eval_one_quadrant(
    quadrant_idx: int,
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For one held-out quadrant, run B repetitions and collect:
      - max-risk on train
      - Risk on test

    Args:
        quadrant_idx (int): Quadrant index
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
        method (str): Minimize the maximum MSE ("mse"), negative reward ("nrw"), or regret ("reg")

    Returns:
        Tuple (main_df, env_metrics_df): Two dataframes with performance metrics
    """

    # def center_per_env(y, env_labels):
    #     y_centered = np.zeros_like(y)
    #     for k in np.unique(env_labels):
    #         mask = env_labels == k
    #         y_centered[mask] = y[mask] - np.mean(y[mask])
    #     return y_centered

    # Masks
    test_mask = env == quadrant_idx
    train_mask = ~test_mask

    X_test, y_test = X[test_mask], y[test_mask]
    X_pool, y_pool = X[train_mask], y[train_mask]
    env_pool = env[train_mask]
    env_test = env[test_mask]

    train_env_indices = np.unique(env_pool)

    if method == "reg":
        rf_regret_te = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf_regret_te.fit(X_test, y_test)
        sols_erm_te = rf_regret_te.predict(X_test)

    # y_test = center_per_env(y_test, env_test)

    main_records = []
    env_metrics_records = []

    for b in tqdm(range(B)):
        # Stratified split (preserve env proportions among the 3 training envs)
        (
            X_tr,
            X_val,
            y_tr,
            y_val,
            env_tr,
            env_val,
        ) = train_test_split(
            X_pool,
            y_pool,
            env_pool,
            test_size=VAL_PERCENTAGE,
            random_state=b,
            stratify=env_pool,
        )

        # y_tr = center_per_env(y_tr, env_tr)
        # y_val = center_per_env(y_val, env_val)

        if method == "reg":
            # Compute the ERM solution in each environment
            sols_erm_tr = np.zeros(env_tr.shape[0])
            sols_erm_val = np.zeros(env_val.shape[0])
            sols_erm_tr_trees = np.zeros((N_ESTIMATORS, env_tr.shape[0]))
            for e in np.unique(env_tr):
                mask_e = env_tr == e
                X_e = X_tr[mask_e]
                y_e = y_tr[mask_e]
                rf_e = RandomForest(
                    "Regression",
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    seed=SEED,
                )
                rf_e.fit(X_e, y_e)
                fitted_e = rf_e.predict(X_e)
                sols_erm_tr[mask_e] = fitted_e
                for i in range(N_ESTIMATORS):
                    fitted_e_tree = rf_e.trees[i].predict(
                        np.ascontiguousarray(X_e.to_numpy())
                    )
                    sols_erm_tr_trees[i, mask_e] = fitted_e_tree
                mask_e_val = env_val == e
                fitted_e_val = rf_e.predict(X_val[mask_e_val])
                sols_erm_val[mask_e_val] = fitted_e_val

        # Fit and predict
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)
        preds_val = rf.predict(X_val)
        preds_test = rf.predict(X_test)

        if method == "mse":
            rf.modify_predictions_trees(env_tr)
        elif method == "nrw":
            rf.modify_predictions_trees(env_tr, method="reward")
        else:
            rf.modify_predictions_trees(
                env_tr,
                method="regret",
                sols_erm=sols_erm_tr,
                sols_erm_trees=sols_erm_tr_trees,
            )
        preds_val_posthoc = rf.predict(X_val)
        preds_test_posthoc = rf.predict(X_test)

        # Compute metrics
        if method == "mse":
            risk_envs_val, max_risk_val = max_mse(
                y_val, preds_val, env_val, ret_ind=True
            )
            risk_val = mean_squared_error(y_val, preds_val)
            risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_val_posthoc, max_risk_val_posthoc = max_mse(
                y_val, preds_val_posthoc, env_val, ret_ind=True
            )
            risk_val_posthoc = mean_squared_error(y_val, preds_val_posthoc)
            risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)

            mse_envs_val = risk_envs_val
            mse_envs_val_posthoc = risk_envs_val_posthoc
        elif method == "nrw":
            risk_envs_val, max_risk_val = min_reward(
                y_val, preds_val, env_val, ret_ind=True
            )
            risk_envs_val = -np.array(risk_envs_val)
            max_risk_val = -max_risk_val
            risk_val = np.nan
            # risk_test = mean_squared_error(y_test, preds_test) - np.mean(
            #     y_test**2
            # )
            risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_val_posthoc, max_risk_val_posthoc = min_reward(
                y_val, preds_val_posthoc, env_val, ret_ind=True
            )
            risk_envs_val_posthoc = -np.array(risk_envs_val_posthoc)
            max_risk_val_posthoc = -max_risk_val_posthoc
            risk_val_posthoc = np.nan
            # risk_test_posthoc = mean_squared_error(
            #     y_test, preds_test_posthoc
            # ) - np.mean(y_test**2)
            risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)

            mse_envs_val, _ = max_mse(y_val, preds_val, env_val, ret_ind=True)
            mse_envs_val_posthoc, _ = max_mse(
                y_val, preds_val_posthoc, env_val, ret_ind=True
            )
        else:
            risk_envs_val, max_risk_val = max_regret(
                y_val, preds_val, sols_erm_val, env_val, ret_ind=True
            )
            risk_val = np.nan
            # risk_test = mean_squared_error(
            #     y_test, preds_test
            # ) - mean_squared_error(y_test, sols_erm_te)
            risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_val_posthoc, max_risk_val_posthoc = max_regret(
                y_val, preds_val_posthoc, sols_erm_val, env_val, ret_ind=True
            )
            risk_val_posthoc = np.nan
            # risk_test_posthoc = mean_squared_error(
            #     y_test, preds_test_posthoc
            # ) - mean_squared_error(y_test, sols_erm_te)
            risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)

            mse_envs_val, _ = max_mse(y_val, preds_val, env_val, ret_ind=True)
            mse_envs_val_posthoc, _ = max_mse(
                y_val, preds_val_posthoc, env_val, ret_ind=True
            )

        for (
            model_name,
            tr,
            te,
            tr_envs,
            tr_pool,
            tr_mse_envs,
        ) in [
            (
                "RF",
                max_risk_val,
                risk_test,
                risk_envs_val,
                risk_val,
                mse_envs_val,
            ),
            (
                f"{NAME_RF}({method})",
                max_risk_val_posthoc,
                risk_test_posthoc,
                risk_envs_val_posthoc,
                risk_val_posthoc,
                mse_envs_val_posthoc,
            ),
        ]:
            # Main performance metrics
            main_records.append(
                {
                    "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                    "Rep": b,
                    "Model": model_name,
                    "Train_max_risk": tr,
                    "Test_risk": te,
                    "Train_risk": tr_pool,
                }
            )

            # Environment-specific performance metrics
            for i, env_value in enumerate(tr_envs):
                env_idx = train_env_indices[i]
                env_metrics_records.append(
                    {
                        "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                        "HeldOutQuadrantIdx": quadrant_idx,
                        "Rep": b,
                        "Model": model_name,
                        "EnvIndex": int(env_idx),
                        "Risk": float(env_value),
                        "MSE": float(tr_mse_envs[i]),
                    }
                )

    return pd.DataFrame.from_records(main_records), pd.DataFrame.from_records(
        env_metrics_records
    )


def gen_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-quadrant-out with train/val/test split

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
        method (str): Minimize the maximum MSE ("mse"), negative reward ("nrw"), or regret ("reg")

    Returns:
        main_result_df (pd.DataFrame): Max risk and results on test data
    """
    # Parallelize over quadrants
    main_dfs = []
    env_metrics_dfs = []
    logger.info("Submitting quadrant tasks to ProcessPoolExecutor")
    with ProcessPoolExecutor() as exe:
        futures = {}
        for qi in range(len(QUADRANTS)):
            logger.info(
                f"Submitting quadrant {QUADRANTS[qi]} (idx={qi}) to pool"
            )
            fut = exe.submit(eval_one_quadrant, qi, X, y, env, method)
            futures[fut] = qi
        for fut in as_completed(futures):
            qi = futures[fut]
            logger.info(
                f"Quadrant {QUADRANTS[qi]} completed, collecting results"
            )
            main_df, env_metrics_df = fut.result()
            main_dfs.append(main_df)
            env_metrics_dfs.append(env_metrics_df)

    # Combine and plot
    main_result_df = pd.concat(main_dfs, ignore_index=True)
    plot_test_risk(
        main_result_df,
        saveplot=True,
        nameplot=f"heldout_mse_{method}",
        method=method,
        out_dir=OUT_DIR,
    )

    env_metrics_result_df = pd.concat(env_metrics_dfs, ignore_index=True)
    plot_envs_risk(
        env_metrics_result_df,
        main_result_df,
        saveplot=True,
        nameplot=f"env_specific_{method}",
        method=method,
        out_dir=OUT_DIR,
    )

    return main_result_df, env_metrics_result_df


def eval_one_quadrant_linear(
    quadrant_idx: int,
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Linear Regression only.
    """
    # Masks
    test_mask = env == quadrant_idx
    train_mask = ~test_mask

    X_test, y_test = X[test_mask], y[test_mask]
    X_pool, y_pool = X[train_mask], y[train_mask]
    env_pool = env[train_mask]

    train_env_indices = np.unique(env_pool)

    main_records = []
    env_metrics_records = []

    for b in tqdm(range(B)):
        # Stratified split across the 3 training envs
        X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
            X_pool,
            y_pool,
            env_pool,
            test_size=VAL_PERCENTAGE,
            random_state=b,
            stratify=env_pool,
        )

        # Fit global ERM Linear Regression
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        preds_val = lr.predict(X_val)
        preds_test = lr.predict(X_test)

        # Compute metrics
        mse_envs_val, max_mse_val = max_mse(
            y_val, preds_val, env_val, ret_ind=True
        )
        mse_test = mean_squared_error(y_test, preds_test)

        # Main performance metrics (Linear Regression only)
        main_records.append(
            {
                "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                "Rep": b,
                "Model": "LR",
                "Train_max_mse": float(max_mse_val),
                "Test_mse": float(mse_test),
            }
        )

        # Environment-specific performance metrics (on validation split)
        for i, env_value in enumerate(mse_envs_val):
            env_idx = train_env_indices[i]
            env_metrics_records.append(
                {
                    "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                    "HeldOutQuadrantIdx": quadrant_idx,
                    "Rep": b,
                    "Model": "LR",
                    "EnvIndex": int(env_idx),
                    "MSE": float(env_value),
                }
            )

    return pd.DataFrame.from_records(main_records), pd.DataFrame.from_records(
        env_metrics_records
    )


def gen_exp_linear(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Linear Regression only.
    """
    main_dfs = []
    env_metrics_dfs = []
    logger.info(
        "Submitting quadrant tasks (Linear Regression) to ProcessPoolExecutor"
    )
    with ProcessPoolExecutor() as exe:
        futures = {}
        for qi in range(len(QUADRANTS)):
            logger.info(
                f"Submitting quadrant {QUADRANTS[qi]} (idx={qi}) to pool"
            )
            fut = exe.submit(eval_one_quadrant_linear, qi, X, y, env)
            futures[fut] = qi

        for fut in as_completed(futures):
            qi = futures[fut]
            logger.info(
                f"Quadrant {QUADRANTS[qi]} completed, collecting results"
            )
            main_df, env_metrics_df = fut.result()
            main_dfs.append(main_df)
            env_metrics_dfs.append(env_metrics_df)

    # Combine and save
    main_result_df = pd.concat(main_dfs, ignore_index=True)
    env_metrics_result_df = pd.concat(env_metrics_dfs, ignore_index=True)

    return main_result_df, env_metrics_result_df


if __name__ == "__main__":
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant(Z)

    print("\nRunning experiment: Minimize maximum MSE across environments\n")
    mse_main_df, mse_envs_df = gen_exp(X, y, env, "mse")

    print(
        "\nRunning experiment: Minimize maximum negative reward across environments\n"
    )
    nrw_main_df, nrw_envs_df = gen_exp(X, y, env, "nrw")

    print(
        "\nRunning experiment: Minimize maximum regret across environments\n"
    )
    reg_main_df, reg_envs_df = gen_exp(X, y, env, "reg")

    combined_df = pd.concat(
        [mse_main_df, nrw_main_df, reg_main_df], ignore_index=True
    )
    plot_test_risk_all_methods(combined_df, saveplot=True, out_dir=OUT_DIR)
    table_df = table_test_risk_all_methods(combined_df)
    latex_str = table_df.to_latex(
        index=False, escape=False, column_format="lcccc"
    )
    with open(
        os.path.join(OUT_DIR, "heldout_mse_all_methods_table.txt"), "w"
    ) as f:
        f.write(latex_str)

    env_all = pd.concat(
        [mse_envs_df, nrw_envs_df, reg_envs_df], ignore_index=True
    )
    plot_envs_mse_all_methods(env_all, saveplot=True, out_dir=OUT_DIR)

    print("\nRunning experiment: Linear Regression\n")
    main_df, env_df = gen_exp_linear(X, y, env)

    write_lr_test_table_txt(main_df, out_dir=OUT_DIR)
    write_lr_env_specific_table_txt(env_df, out_dir=OUT_DIR)
