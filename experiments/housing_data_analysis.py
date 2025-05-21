import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from nldg.utils import max_mse
from adaXT.random_forest import RandomForest
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "..", "data")
results_dir = os.path.join(script_dir, "..", "results")
os.makedirs(results_dir, exist_ok=True)
out_data_dir = os.path.join(results_dir, "output_data_housing_rf")
os.makedirs(out_data_dir, exist_ok=True)
RESULTS_PATH = out_data_dir
QUADRANTS = ["SW", "SE", "NW", "NE"]
B = 1
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
    data_path = os.path.join(DATA_PATH, "housing-sklearn.csv")
    dat = pd.read_csv(filepath_or_buffer=data_path)

    y = dat["MedHouseVal"]
    Z = dat[["Latitude", "Longitude"]]
    X = dat.drop(["MedHouseVal", "Latitude", "Longitude"], axis=1)

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    main_records = []
    env_metrics_records = []

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

        rf.modify_predictions_trees(env_tr)
        preds_tr_minmax = rf.predict(X_tr)
        preds_val_minmax = rf.predict(X_val)
        preds_test_minmax = rf.predict(X_test)

        # Compute metrics
        mse_envs_tr, max_mse_tr = max_mse(y_tr, preds_tr, env_tr, ret_ind=True)
        mse_envs_val, max_mse_val = max_mse(
            y_val, preds_val, env_val, ret_ind=True
        )
        max_mse_test = max_mse(y_test, preds_test, env_test)

        mse_envs_tr_minmax, max_mse_tr_minmax = max_mse(
            y_tr, preds_tr_minmax, env_tr, ret_ind=True
        )
        mse_envs_val_minmax, max_mse_val_minmax = max_mse(
            y_val, preds_val_minmax, env_val, ret_ind=True
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
            # Main performance metrics
            main_records.append(
                {
                    "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                    "Rep": b,
                    "Model": model_name,
                    "Train_maxMSE": tr,
                    "Val_maxMSE": va,
                    "Test_MSE": te,
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
                        "DataSplit": "Train",
                        "MSE": float(env_value),
                    }
                )

            for i, env_value in enumerate(va_envs):
                env_idx = train_env_indices[i]
                env_metrics_records.append(
                    {
                        "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                        "HeldOutQuadrantIdx": quadrant_idx,
                        "Rep": b,
                        "Model": model_name,
                        "EnvIndex": int(env_idx),
                        "DataSplit": "Val",
                        "MSE": float(env_value),
                    }
                )

    return pd.DataFrame.from_records(main_records), pd.DataFrame.from_records(
        env_metrics_records
    )


def main():
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant_env(Z)

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
            fut = exe.submit(eval_one_quadrant, qi, X, y, env)
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
    main_out_path = os.path.join(RESULTS_PATH, "main_metrics.csv")
    main_result_df.to_csv(main_out_path, index=False)
    logger.info(f"Saved main results to {main_out_path}")

    env_metrics_result_df = pd.concat(env_metrics_dfs, ignore_index=True)
    env_metrics_out_path = os.path.join(
        RESULTS_PATH, "env_specific_metrics.csv"
    )
    env_metrics_result_df.to_csv(env_metrics_out_path, index=False)
    logger.info(f"Saved environment metrics to {env_metrics_out_path}")


if __name__ == "__main__":
    main()
