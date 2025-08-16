import os
import copy
import logging
import numpy as np
import pandas as pd
from nldg.utils import max_mse, max_regret, min_reward
from adaXT.random_forest import RandomForest
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import (
    plot_max_risk_vs_hyperparam,
)
from ca_housing_analysis import assign_quadrant, load_data

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
CA_DIR = os.path.join(RESULTS_DIR, "output_ca_housing")
os.makedirs(CA_DIR, exist_ok=True)
OUT_DIR = os.path.join(CA_DIR, "hyperparameters")
os.makedirs(OUT_DIR, exist_ok=True)

QUADRANTS = ["Env 1", "Env 2", "Env 3", "Env 4"]
B = 20
VAL_PERCENTAGE = 0.3
N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
N = 20

NAME_RF = "MaxRM-RF"


def hyperparam_exp(
    X: pd.DataFrame, y: pd.Series, env: np.ndarray, hyperparam: str
) -> None:
    """
    Compares RF with the variants that minimize the maximum risk
    across environments when a certain hyperparameter varies.
    Confidence intervals are constructed using resampling.

    Args:
        X (pd.DataFrame): Feature matrix (n, p)
        y (pd.Series):   Target vector (n,)
        env (np.ndarray): Environment labels (n,)
        hyperparam (str): Hyperparameter name
    """
    # grid of hyperparameters
    if hyperparam == "mtry":
        hyperparam_values = np.arange(1, X.shape[1] + 1)
    elif hyperparam == "min_samples_leaf":
        hyperparam_values = np.array([10, 15, 20, 25, 30, 35])
    else:
        hyperparam_values = np.array([5, 6, 7, 8, 9, 10])
    len_grid = len(hyperparam_values)

    results_rf_mse, results_rf_nrw, results_rf_reg = (
        np.zeros((N, len_grid)),
        np.zeros((N, len_grid)),
        np.zeros((N, len_grid)),
    )
    results_mse, results_nrw, results_reg = (
        np.zeros((N, len_grid)),
        np.zeros((N, len_grid)),
        np.zeros((N, len_grid)),
    )

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

        for j, param in enumerate(hyperparam_values):
            # Fit standard RF for each environment separately
            # This is used for the regret
            sols_erm = np.zeros(env_tr.shape[0])
            sols_erm_val = np.zeros(env_val.shape[0])
            sols_erm_trees = np.zeros((N_ESTIMATORS, env_tr.shape[0]))
            for e in np.unique(env_tr):
                mask = env_tr == e
                X_e = X_tr[mask]
                Y_e = y_tr[mask]
                if hyperparam == "mtry":
                    rf_e = RandomForest(
                        "Regression",
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        seed=SEED,
                        max_features=int(param),
                    )
                elif hyperparam == "min_samples_leaf":
                    rf_e = RandomForest(
                        "Regression",
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=int(param),
                        seed=SEED,
                    )
                else:
                    rf_e = RandomForest(
                        "Regression",
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        seed=SEED,
                        max_depth=int(param),
                    )
                rf_e.fit(X_e, Y_e)
                fitted_e = rf_e.predict(X_e)
                sols_erm[mask] = fitted_e
                for k in range(N_ESTIMATORS):
                    fitted_e_tree = rf_e.trees[k].predict(
                        np.ascontiguousarray(X_e.to_numpy())
                    )
                    sols_erm_trees[k, mask] = fitted_e_tree
                mask_e_val = env_val == e
                fitted_e_val = rf_e.predict(X_val[mask_e_val])
                sols_erm_val[mask_e_val] = fitted_e_val

            # RF
            if hyperparam == "mtry":
                rf = RandomForest(
                    "Regression",
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    seed=SEED,
                    max_features=int(param),
                )
            elif hyperparam == "min_samples_leaf":
                rf = RandomForest(
                    "Regression",
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=int(param),
                    seed=SEED,
                )
            else:
                rf = RandomForest(
                    "Regression",
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    seed=SEED,
                    max_depth=int(param),
                )
            rf.fit(X_tr, y_tr)
            fitted_rf = rf.predict(X_val)
            results_rf_mse[i, j] = max_mse(y_val, fitted_rf, env_val)
            results_rf_nrw[i, j] = -min_reward(y_val, fitted_rf, env_val)
            results_rf_reg[i, j] = max_regret(
                y_val, fitted_rf, sols_erm_val, env_val
            )

            # min max MSE
            rf_mse = copy.deepcopy(rf)
            rf_mse.modify_predictions_trees(env_tr, solver="ECOS")
            fitted_mse = rf_mse.predict(X_val)
            results_mse[i, j] = max_mse(y_val, fitted_mse, env_val)

            # min max Negative Reward
            rf_nrw = copy.deepcopy(rf)
            rf_nrw.modify_predictions_trees(
                env_tr, method="reward", solver="ECOS"
            )
            fitted_nrw = rf_nrw.predict(X_val)
            results_nrw[i, j] = -min_reward(y_val, fitted_nrw, env_val)

            # min max Regret
            rf_reg = copy.deepcopy(rf)
            rf_reg.modify_predictions_trees(
                env_tr,
                method="regret",
                sols_erm=sols_erm,
                sols_erm_trees=sols_erm_trees,
                solver="ECOS",
            )
            fitted_reg = rf_reg.predict(X_val)
            results_reg[i, j] = max_regret(
                y_val, fitted_reg, sols_erm_val, env_val
            )

    def plot_df(df_base, df_posthoc, nameplot, suffix):
        df_base["method"] = "RF"
        df_posthoc["method"] = f"{NAME_RF}({suffix})"
        df_long = pd.concat(
            [
                df_base.melt(
                    id_vars="method", var_name=hyperparam, value_name="risk"
                ),
                df_posthoc.melt(
                    id_vars="method", var_name=hyperparam, value_name="risk"
                ),
            ],
            ignore_index=True,
        )
        df_long[hyperparam] = df_long[hyperparam].astype(int)
        plot_max_risk_vs_hyperparam(
            df_long,
            hyperparam=hyperparam,
            saveplot=True,
            nameplot=nameplot,
            out_dir=OUT_DIR,
            suffix=suffix,
        )

    # MSE
    df_rf_mse = pd.DataFrame(results_rf_mse, columns=hyperparam_values)
    df_mse = pd.DataFrame(results_mse, columns=hyperparam_values)
    plot_df(df_rf_mse, df_mse, f"max_mse_{hyperparam}", "mse")

    # Negative Reward
    df_rf_nrw = pd.DataFrame(results_rf_nrw, columns=hyperparam_values)
    df_nrw = pd.DataFrame(results_nrw, columns=hyperparam_values)
    plot_df(df_rf_nrw, df_nrw, f"max_nrw_{hyperparam}", "nrw")

    # Regret
    df_rf_reg = pd.DataFrame(results_rf_reg, columns=hyperparam_values)
    df_reg = pd.DataFrame(results_reg, columns=hyperparam_values)
    plot_df(df_rf_reg, df_reg, f"max_reg_{hyperparam}", "reg")

    logger.info(f"Saved metrics to {OUT_DIR}")


if __name__ == "__main__":
    logger.info("Loading data and assigning environments")
    X, y, Z = load_data()
    env = assign_quadrant(Z)

    logger.info("Running mtry experiment")
    hyperparam_exp(X, y, env, "mtry")

    logger.info("Running min_samples_leaf experiment")
    hyperparam_exp(X, y, env, "min_samples_leaf")

    logger.info("Running max_depth experiment")
    hyperparam_exp(X, y, env, "max_depth")
