import os
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from nldg.utils import max_mse
from adaXT.random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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
B = 20
VAL_PERCENTAGE = 0.2
N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
M = 200
N = 20


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


# def assign_quadrant_env(
#     Z: pd.DataFrame,
# ) -> np.ndarray:
#     """
#     Creates the environment label based on geographic criteria.
#
#     Args:
#         Z (pd.DataFrame): Additional covariates (Latitude, Longitude)
#
#     Returns:
#         env (np.ndarray): Environment label
#     """
#     lat, lon = Z["Latitude"], Z["Longitude"]
#     # Setting 1: 35, -120
#     # Setting 2: 36, -119.8
#     north = lat >= 36
#     south = ~north
#     east = lon >= -119.8
#     west = ~east
#
#     env = np.zeros(len(Z), dtype=int)
#     env[south & west] = 0  # SW
#     env[south & east] = 1  # SE
#     env[north & west] = 2  # NW
#     env[north & east] = 3  # NE
#     return env


# Setting 3
def assign_quadrant(Z: pd.DataFrame) -> np.ndarray:
    """
    Creates the environment label based on geographic criteria.

    Args:
        Z (pd.DataFrame): Additional covariates (Latitude, Longitude)

    Returns:
        env (np.ndarray): Environment label
    """
    lat, lon = Z["Latitude"], Z["Longitude"]

    west = lon < -121
    east = ~west

    # For west side: split at 38
    sw = (lat < 38) & west
    nw = (lat >= 38) & west

    # For east side: split at 36
    se = (lat < 36) & east
    ne = (lat >= 36) & east

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For one held-out quadrant, run B repetitions and collect:
      - max-MSE on train
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

    for b in tqdm(range(B)):
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
        preds_test = rf.predict(X_test)

        rf.modify_predictions_trees(env_tr)
        preds_tr_minmax = rf.predict(X_tr)
        preds_test_minmax = rf.predict(X_test)

        # Compute metrics
        mse_envs_tr, max_mse_tr = max_mse(y_tr, preds_tr, env_tr, ret_ind=True)
        mse_tr = mean_squared_error(y_tr, preds_tr)
        max_mse_test = max_mse(y_test, preds_test, env_test)

        mse_envs_tr_minmax, max_mse_tr_minmax = max_mse(
            y_tr, preds_tr_minmax, env_tr, ret_ind=True
        )
        mse_tr_minmax = mean_squared_error(y_tr, preds_tr_minmax)
        max_mse_test_minmax = max_mse(y_test, preds_test_minmax, env_test)

        for model_name, tr, te, tr_envs, tr_pool in [
            (
                "RF",
                max_mse_tr,
                max_mse_test,
                mse_envs_tr,
                mse_tr,
            ),
            (
                "Post-RF",
                max_mse_tr_minmax,
                max_mse_test_minmax,
                mse_envs_tr_minmax,
                mse_tr_minmax,
            ),
        ]:
            # Main performance metrics
            main_records.append(
                {
                    "HeldOutQuadrant": QUADRANTS[quadrant_idx],
                    "Rep": b,
                    "Model": model_name,
                    "Train_maxMSE": tr,
                    "Test_MSE": te,
                    "Train_MSE": tr_pool,
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
                        "MSE": float(env_value),
                    }
                )

    return pd.DataFrame.from_records(main_records), pd.DataFrame.from_records(
        env_metrics_records
    )


def run_ttv_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Leave-one-quadrant-out with train/val/test split

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
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
    main_out_path = os.path.join(RESULTS_PATH, "maxmse_ttv.csv")
    main_result_df.to_csv(main_out_path, index=False)
    logger.info(f"Saved main results to {main_out_path}")

    env_metrics_result_df = pd.concat(env_metrics_dfs, ignore_index=True)
    env_metrics_out_path = os.path.join(RESULTS_PATH, "env_specific_mse.csv")
    env_metrics_result_df.to_csv(env_metrics_out_path, index=False)
    logger.info(f"Saved environment metrics to {env_metrics_out_path}")


def run_t_bootstrap_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Bootstrap on full dataset

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
    """
    n = len(y)
    rng = np.random.default_rng(SEED)

    # Fit once on full data for point estimates
    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf.fit(X, y)
    p_full = rf.predict(X)
    rf.modify_predictions_trees(env)
    p_full_mm = rf.predict(X)

    # Compute theta_hat per quadrant
    theta = {}
    for name, preds in [("RF", p_full), ("Post-RF", p_full_mm)]:
        theta[name] = [
            ((y[env == q] - preds[env == q]) ** 2).mean() for q in range(4)
        ]

    # Bootstrap replicates
    boot = {("RF", q): [] for q in range(4)}
    boot.update({("Post-RF", q): [] for q in range(4)})

    for m in tqdm(range(M)):
        idxs = rng.choice(n, size=n, replace=True)
        Xb, yb, eb = X.iloc[idxs], y.iloc[idxs], env[idxs]
        rf_b = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf_b.fit(Xb, yb)
        pb = rf_b.predict(Xb)
        rf_b.modify_predictions_trees(eb)
        pb_mm = rf_b.predict(Xb)

        for name, preds in [("RF", pb), ("Post-RF", pb_mm)]:
            for q in range(4):
                mask = eb == q
                if mask.sum():
                    boot[(name, q)].append(
                        ((yb[mask] - preds[mask]) ** 2).mean()
                    )

    # Build result DataFrame
    records = []
    for name in ["RF", "Post-RF"]:
        for q in range(4):
            arr = np.array(boot[(name, q)])
            mean_hat = theta[name][q]
            q_low, q_high = np.percentile(arr, [2.5, 97.5])
            mean_boot = np.mean(arr)
            std_boot = np.std(arr, ddof=1)
            records.append(
                {
                    "Method": name,
                    "Quadrant": QUADRANTS[q],
                    "MSE_mean": mean_hat,
                    "Lower_CI_basic": 2 * mean_hat - q_high,
                    "Upper_CI_basic": 2 * mean_hat - q_low,
                    "Lower_CI_perc": q_low,
                    "Upper_CI_perc": q_high,
                    "Lower_CI_normal": 2 * mean_hat
                    - mean_boot
                    - 1.96 * std_boot,
                    "Upper_CI_normal": 2 * mean_hat
                    - mean_boot
                    + 1.96 * std_boot,
                }
            )

    df_boot = pd.DataFrame(records)

    path = os.path.join(RESULTS_PATH, "env_specific_mse_bootstrap.csv")
    df_boot.to_csv(path, index=False)
    logger.info(f"Saved bootstrap metrics to {path}")


def run_t_resample_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Resample the full dataset

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
    """
    results_rf = np.zeros((N, len(QUADRANTS) + 1))
    results_minmax = np.zeros((N, len(QUADRANTS) + 1))
    for i in tqdm(range(N)):
        X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
            X,
            y,
            env,
            test_size=VAL_PERCENTAGE,
            random_state=i,
            stratify=env,
        )

        # Fit and predict
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)
        fitted_tr = rf.predict(X_tr)

        rf.modify_predictions_trees(env_tr)
        fitted_tr_minmax = rf.predict(X_tr)

        # Compute metrics
        mse_envs_tr, max_mse_tr = max_mse(
            y_tr, fitted_tr, env_tr, ret_ind=True
        )
        mse_tr = mean_squared_error(y_tr, fitted_tr)
        mse_envs_tr_minmax, max_mse_tr_minmax = max_mse(
            y_tr, fitted_tr_minmax, env_tr, ret_ind=True
        )
        mse_tr_minmax = mean_squared_error(y_tr, fitted_tr_minmax)

        results_rf[i, : len(QUADRANTS)] = mse_envs_tr
        results_rf[i, len(QUADRANTS)] = mse_tr
        results_minmax[i, : len(QUADRANTS)] = mse_envs_tr_minmax
        results_minmax[i, len(QUADRANTS)] = mse_tr_minmax

    names = QUADRANTS + ["Overall"]
    df_rf = pd.DataFrame(results_rf, columns=names)
    df_rf["method"] = "RF"

    df_mm = pd.DataFrame(results_minmax, columns=names)
    df_mm["method"] = "Post-RF"

    df_all = pd.concat([df_rf, df_mm], ignore_index=True)

    path = os.path.join(RESULTS_PATH, "env_specific_mse_resample.csv")
    df_all.to_csv(path, index=False)
    logger.info(f"Saved resampling metrics to {path}")


def run_mtry_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Comparison of RF and Post-RF with different mtry

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        env (np.ndarray): Environment labels
    """
    mtry = np.arange(1, X.shape[1] + 1)
    results = dict()
    for m in tqdm(mtry):
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            max_features=int(m),
        )
        rf.fit(X, y)
        fitted_rf = rf.predict(X)

        rf.modify_predictions_trees(env)
        fitted_minmax = rf.predict(X)

        max_mse_rf = max_mse(np.array(y), fitted_rf, env)
        max_mse_minmax = max_mse(np.array(y), fitted_minmax, env)

        results[m] = [max_mse_rf, max_mse_minmax]

    df = pd.DataFrame.from_dict(
        results, orient="index", columns=["maxMSE_RF", "maxMSE_Post-RF"]
    )
    df = df.reset_index().rename(columns={"index": "mtry"})
    df["mtry"] = df["mtry"].astype(int)

    path = os.path.join(RESULTS_PATH, "maxmse_mtry.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved resampling metrics to {path}")


def run_mtry_resample_exp(
    X: pd.DataFrame,
    y: pd.Series,
    env: np.ndarray,
) -> None:
    """
    Comparison of RF and Post-RF with different mtry,
    repeated N times on stratified resamples of the data.

    Args:
        X (pd.DataFrame): Feature matrix (shape n×p)
        y (pd.Series):   Target vector (length n)
        env (np.ndarray): Environment labels (length n)
    """
    # grid of mtry values
    mtry_values = np.arange(1, X.shape[1] + 1)
    p = len(mtry_values)

    results_rf = np.zeros((N, p))
    results_mm = np.zeros((N, p))

    for i in tqdm(range(N)):
        X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
            X,
            y,
            env,
            test_size=VAL_PERCENTAGE,
            random_state=i,
            stratify=env,
        )

        for j, m in enumerate(mtry_values):
            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                max_features=int(m),
            )
            rf.fit(X_tr, y_tr)
            fitted_rf = rf.predict(X_tr)

            # Post-RF
            rf.modify_predictions_trees(env_tr)
            fitted_mm = rf.predict(X_tr)

            # compute worst‐case (max) MSE over environments
            results_rf[i, j] = max_mse(np.array(y_tr), fitted_rf, env_tr)
            results_mm[i, j] = max_mse(np.array(y_tr), fitted_mm, env_tr)

    # build a long DataFrame for easy analysis / plotting
    # first, wide → two DataFrames with shape (N×p)
    df_rf = pd.DataFrame(results_rf, columns=mtry_values)
    df_rf["method"] = "RF"
    df_mm = pd.DataFrame(results_mm, columns=mtry_values)
    df_mm["method"] = "Post-RF"

    # melt both into long form
    df_rf_long = df_rf.melt(
        id_vars="method", var_name="mtry", value_name="maxMSE"
    )
    df_mm_long = df_mm.melt(
        id_vars="method", var_name="mtry", value_name="maxMSE"
    )

    df_all = pd.concat([df_rf_long, df_mm_long], ignore_index=True)
    df_all["mtry"] = df_all["mtry"].astype(int)

    path = os.path.join(RESULTS_PATH, "maxmse_mtry_resample.csv")
    df_all.to_csv(path, index=False)
    logger.info(f"Saved resampling metrics to {path}")


def main(version: str):
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant_env(Z)

    if version == "train_test_val":
        # Divide into train and test.
        # The train data is further divided into train and validation
        run_ttv_exp(X, y, env)

    elif version == "train_bootstrap":
        # The whole dataset is used to fit the models.
        # Confidence intervals for the MSE are constructed using bootstrap.
        run_t_bootstrap_exp(X, y, env)

    elif version == "train_resample":
        # The whole dataset is used to fit the models.
        # Confidence intervals for the MSE are constructed using resampling.
        run_t_resample_exp(X, y, env)

    elif version == "train_mtry":
        # The whole dataset is used to fit the models.
        # Comparison of the methods in terms of mtry.
        run_mtry_exp(X, y, env)

    elif version == "train_mtry_resample":
        # The whole dataset is used to fit the models.
        # Comparison of the methods in terms of mtry.
        # Confidence intervals are constructred using resampling.
        run_mtry_resample_exp(X, y, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RF experiments on California housing."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="train_test_val",
        choices=[
            "train_test_val",
            "train_bootstrap",
            "train_resample",
            "train_mtry",
            "train_mtry_resample",
        ],
        help="Experiment version (default: 'train_test_val'). "
        "Must be one of 'train_test_val', 'train_bootstrap', 'train_resample', 'train_mtry' or 'train_mtry_resample'.",
    )
    args = parser.parse_args()
    main(args.version)
