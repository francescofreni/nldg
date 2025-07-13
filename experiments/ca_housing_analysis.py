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
from tqdm import tqdm
from utils import (
    plot_max_mse_mtry,
    plot_test_risk,
    plot_envs_risk,
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

QUADRANTS = ["SW", "SE", "NW", "NE"]
B = 20
VAL_PERCENTAGE = 0.2
N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
N = 20

NAME_RF = "WORME-RF"


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
    north = lat >= 35
    south = ~north
    east = lon >= -120
    west = ~east

    env = np.zeros(len(Z), dtype=int)
    env[south & west] = 0  # SW
    env[south & east] = 1  # SE
    env[north & west] = 2  # NW
    env[north & east] = 3  # NE
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

    if method == "nrw":
        y_test = y_test - np.mean(y_test)

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

        if method == "reg":
            # Compute the ERM solution in each environment
            sols_erm_tr = np.zeros(env_tr.shape[0])
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
                    fitted_e_tree = rf_e.trees[i].predict(X_e.to_numpy())
                    sols_erm_tr_trees[i, mask_e] = fitted_e_tree

        if method == "nrw":
            y_tr_demean = np.zeros_like(y_tr)
            for env in np.unique(env_tr):
                mask = env_tr == env
                y_tr_e = y_tr[mask]
                y_tr_demean[mask] = y_tr_e - np.mean(y_tr_e)
            y_tr = y_tr_demean

        # Fit and predict
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)
        preds_tr = rf.predict(X_tr)
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
        preds_tr_posthoc = rf.predict(X_tr)
        preds_test_posthoc = rf.predict(X_test)

        # Compute metrics
        if method == "mse":
            risk_envs_tr, max_risk_tr = max_mse(
                y_tr, preds_tr, env_tr, ret_ind=True
            )
            risk_tr = mean_squared_error(y_tr, preds_tr)
            risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_tr_posthoc, max_risk_tr_posthoc = max_mse(
                y_tr, preds_tr_posthoc, env_tr, ret_ind=True
            )
            risk_tr_posthoc = mean_squared_error(y_tr, preds_tr_posthoc)
            risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)
        elif method == "nrw":
            risk_envs_tr, max_risk_tr = min_reward(
                y_tr, preds_tr, env_tr, ret_ind=True
            )
            risk_envs_tr = -np.array(risk_envs_tr)
            max_risk_tr = -max_risk_tr
            risk_tr = np.nan
            risk_test = mean_squared_error(y_test, preds_test) - np.mean(
                y_test**2
            )
            # risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_tr_posthoc, max_risk_tr_posthoc = min_reward(
                y_tr, preds_tr_posthoc, env_tr, ret_ind=True
            )
            risk_envs_tr_posthoc = -np.array(risk_envs_tr_posthoc)
            max_risk_tr_posthoc = -max_risk_tr_posthoc
            risk_tr_posthoc = np.nan
            risk_test_posthoc = mean_squared_error(
                y_test, preds_test_posthoc
            ) - np.mean(y_test**2)
            # risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)
        else:
            risk_envs_tr, max_risk_tr = max_regret(
                y_tr, preds_tr, sols_erm_tr, env_tr, ret_ind=True
            )
            risk_tr = np.nan
            risk_test = mean_squared_error(
                y_test, preds_test
            ) - mean_squared_error(y_test, sols_erm_te)
            # risk_test = mean_squared_error(y_test, preds_test)

            risk_envs_tr_posthoc, max_risk_tr_posthoc = max_regret(
                y_tr, preds_tr_posthoc, sols_erm_tr, env_tr, ret_ind=True
            )
            risk_tr_posthoc = np.nan
            risk_test_posthoc = mean_squared_error(
                y_test, preds_test_posthoc
            ) - mean_squared_error(y_test, sols_erm_te)
            # risk_test_posthoc = mean_squared_error(y_test, preds_test_posthoc)

        for (
            model_name,
            tr,
            te,
            tr_envs,
            tr_pool,
        ) in [
            (
                "RF",
                max_risk_tr,
                risk_test,
                risk_envs_tr,
                risk_tr,
            ),
            (
                f"{NAME_RF}(posthoc-{method})",
                max_risk_tr_posthoc,
                risk_test_posthoc,
                risk_envs_tr_posthoc,
                risk_tr_posthoc,
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
) -> None:
    """
    Leave-one-quadrant-out with train/val/test split

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
        method (str): Minimize the maximum MSE ("mse"), negative reward ("nrw"), or regret ("reg")
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
        nameplot=f"heldout_{method}",
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


def mtry_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Compares RF with the variants that minimize the maximum risk
    across environments when mtry varies.
    Confidence intervals are constructed using resampling.

    Args:
        X (pd.DataFrame): Feature matrix (n, p)
        y (pd.Series):   Target vector (n,)
        env (np.ndarray): Environment labels (n,)
    """
    # grid of mtry values
    mtry_values = np.arange(1, X.shape[1] + 1)
    p = len(mtry_values)

    results_rf_mse, results_rf_nrw, results_rf_reg = np.zeros((N, p))
    results_mse, results_nrw, results_reg = np.zeros((N, p))

    for i in tqdm(range(N)):
        X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
            X,
            y,
            env,
            test_size=VAL_PERCENTAGE,
            random_state=i,
            stratify=env,
        )
        y_tr = np.array(y_tr)
        env_tr = np.array(env_tr)

        for j, m in enumerate(mtry_values):
            # Fit standard RF for each environment separately
            # This is used for the regret
            sols_erm = np.zeros(env_tr.shape[0])
            sols_erm_trees = np.zeros((N_ESTIMATORS, env_tr.shape[0]))
            for env in np.unique(env_tr):
                mask = env_tr == env
                X_e = X_tr[mask]
                Y_e = y_tr[mask]
                rf_e = RandomForest(
                    "Regression",
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    seed=SEED,
                    max_features=int(m),
                )
                rf_e.fit(X_e, Y_e)
                fitted_e = rf_e.predict(X_e)
                sols_erm[mask] = fitted_e
                for i in range(N_ESTIMATORS):
                    fitted_e_tree = rf_e.trees[i].predict(X_e)
                    sols_erm_trees[i, mask] = fitted_e_tree

            # RF
            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                max_features=int(m),
            )
            rf.fit(X_tr, y_tr)
            fitted_rf = rf.predict(X_tr)
            results_rf_mse[i, j] = max_mse(y_tr, fitted_rf, env_tr)
            results_rf_nrw[i, j] = -min_reward(y_tr, fitted_rf, env_tr)
            results_rf_reg[i, j] = max_regret(
                y_tr, fitted_rf, sols_erm, env_tr
            )

            # min max MSE
            rf_mse = copy.deepcopy(rf)
            rf_mse.modify_predictions_trees(env_tr)
            fitted_mse = rf_mse.predict(X_tr)
            results_mse[i, j] = max_mse(y_tr, fitted_mse, env_tr)

            # min max Negative Reward
            rf_nrw = copy.deepcopy(rf)
            rf_nrw.modify_predictions_trees(env_tr, method="reward")
            fitted_nrw = rf_nrw.predict(X_tr)
            results_nrw[i, j] = -min_reward(y_tr, fitted_nrw, env_tr)

            # min max Regret
            rf_reg = copy.deepcopy(rf)
            rf_reg.modify_predictions_trees(
                env_tr,
                method="regret",
                sols_erm=sols_erm,
                sols_erm_trees=sols_erm_trees,
            )
            fitted_reg = rf_reg.predict(X_tr)
            results_reg[i, j] = max_regret(y_tr, fitted_reg, sols_erm, env_tr)

    def plot_df(df_base, df_posthoc, nameplot, suffix):
        df_base["method"] = "RF"
        df_posthoc["method"] = f"{NAME_RF}(posthoc-{suffix})"
        df_long = pd.concat(
            [
                df_base.melt(
                    id_vars="method", var_name="mtry", value_name="risk"
                ),
                df_posthoc.melt(
                    id_vars="method", var_name="mtry", value_name="risk"
                ),
            ],
            ignore_index=True,
        )
        df_long["mtry"] = df_long["mtry"].astype(int)
        plot_max_mse_mtry(
            df_long,
            saveplot=True,
            nameplot=nameplot,
            out_dir=OUT_DIR,
            suffix=suffix,
        )

    # MSE
    df_rf_mse = pd.DataFrame(results_rf_mse, columns=mtry_values)
    df_mse = pd.DataFrame(results_mse, columns=mtry_values)
    plot_df(df_rf_mse, df_mse, "max_mse_mtry", "mse")

    # Negative Reward
    df_rf_nrw = pd.DataFrame(results_rf_nrw, columns=mtry_values)
    df_nrw = pd.DataFrame(results_nrw, columns=mtry_values)
    plot_df(df_rf_nrw, df_nrw, "max_nrw_mtry", "nrw")

    # Regret
    df_rf_reg = pd.DataFrame(results_rf_reg, columns=mtry_values)
    df_reg = pd.DataFrame(results_reg, columns=mtry_values)
    plot_df(df_rf_reg, df_reg, "max_reg_mtry", "reg")

    logger.info(f"Saved metrics to {OUT_DIR}")


if __name__ == "__main__":
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant(Z)

    print("\nRunning experiment: Minimize maximum MSE across environments\n")
    gen_exp(X, y, env, "mse")

    print(
        "\nRunning experiment: Minimize maximum negative reward across environments\n"
    )
    gen_exp(X, y, env, "nrw")

    print(
        "\nRunning experiment: Minimize maximum regret across environments\n"
    )
    gen_exp(X, y, env, "reg")

    print("\nRunning mtry experiment:\n")
    mtry_exp(X, y, env)
