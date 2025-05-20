import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from nldg.utils import max_mse
from adaXT.random_forest import RandomForest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUADRANTS = ["SW", "SE", "NW", "NE"]
B = 20
VAL_PERCENTAGE = 0.2
N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
RESULTS_FOLDER = "results"


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and preprocess the California housing dataset.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        Z (pd.DataFrame): Additional covariates
    """
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    Z = X[["Latitude", "Longitude"]]
    X = X.drop(["Latitude", "Longitude"], axis=1)
    return X, y, Z


def assign_quadrant_env(
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
) -> pd.DataFrame:
    """
    For one held-out quadrant, run B repetitions and collect:
      - max-MSE on train
      - max-MSE on val
      - MSE on test

    Args:
        quadrant_idx (int): Quadrant index
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
    """
    logger.info(
        f"Starting evaluation for quadrant {QUADRANTS[quadrant_idx]} (idx={quadrant_idx})"
    )

    # Masks
    test_mask = env == quadrant_idx
    train_mask = ~test_mask

    X_test, y_test = X[test_mask], y[test_mask]
    X_pool, y_pool = X[train_mask], y[train_mask]
    env_pool = env[train_mask]
    env_test = env[test_mask]

    records = []
    for b in range(B):
        # Stratified split (preserve env proportions among the 3 training envs)
        X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
            X_pool,
            y_pool,
            env_pool,
            test_size=VAL_PERCENTAGE,
            random_state=b,
            stratify=env_pool,
        )

        # Fit and predict
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)
        preds_tr = rf.predict(X_tr)
        preds_val = rf.predict(X_val)
        preds_test = rf.predict(X_test)

        rf.modify_predictions_trees(env_pool)
        preds_tr_minmax = rf.predict(X_tr)
        preds_val_minmax = rf.predict(X_val)
        preds_test_minmax = rf.predict(X_test)

        # Compute metrics
        mse_envs_tr, max_mse_tr = max_mse(
            y_tr, preds_tr, env_pool, ret_ind=True
        )
        mse_envs_val, max_mse_val = max_mse(
            y_val, preds_val, env_pool, ret_ind=True
        )
        max_mse_test = max_mse(y_test, preds_test, env_test)

        mse_envs_tr_minmax, max_mse_tr_minmax = max_mse(
            y_tr, preds_tr_minmax, env_pool, ret_ind=True
        )
        mse_envs_val_minmax, max_mse_val_minmax = max_mse(
            y_val, preds_val_minmax, env_pool, ret_ind=True
        )
        max_mse_test_minmax = max_mse(y_test, preds_test_minmax, env_test)

        for model_name, tr, va, te, tr_envs, va_envs in [
            (
                "RF",
                max_mse_tr,
                max_mse_val,
                max_mse_test,
                mse_envs_tr,
                mse_envs_val,
            ),
            (
                "MinMaxRF",
                max_mse_tr_minmax,
                max_mse_val_minmax,
                max_mse_test_minmax,
                mse_envs_tr_minmax,
                mse_envs_val_minmax,
            ),
        ]:
            tr_envs_1, tr_envs_2, tr_envs_3 = tr_envs
            va_envs_1, va_envs_2, va_envs_3 = va_envs
            records.append(
                {
                    "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                    "Rep": b,
                    "Model": model_name,
                    "Train_maxMSE": tr,
                    "Val_maxMSE": va,
                    "Test_MSE": te,
                    "Train_env1_MSE": tr_envs_1,
                    "Train_env2_MSE": tr_envs_2,
                    "Train_env3_MSE": tr_envs_3,
                    "Val_env1_MSE": va_envs_1,
                    "Val_env2_MSE": va_envs_2,
                    "Val_env3_MSE": va_envs_3,
                }
            )

    logger.info(f"Finished quadrant {QUADRANTS[quadrant_idx]}")
    return pd.DataFrame.from_records(records)


def main():
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant_env(Z)

    # Parallelize over quadrants
    dfs = []
    logger.info("Submitting quadrant tasks to ProcessPoolExecutor")
    with ProcessPoolExecutor() as exe:
        futures = {}
        for qi in range(len(QUADRANTS)):
            logger.info(
                f"Submitting quadrant {QUADRANTS[qi]} (idx={qi}) to pool"
            )
            fut = exe.submit(eval_one_quadrant, qi, X, y, env)
            futures[fut] = qi
        for fut in as_completed(futures):
            qi = futures[fut]
            logger.info(
                f"Quadrant {QUADRANTS[qi]} completed, collecting results"
            )
            dfs.append(fut.result())

    # Combine and save
    result_df = pd.concat(dfs, ignore_index=True)

    results_dir = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "real_word_experiment.csv")
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
