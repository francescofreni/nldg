import argparse
import logging
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error
from adaXT.random_forest import RandomForest
from dataloader import generate_fold_info, get_fold_df
from fluxnet.eval import evaluate_fold
from nldg.utils import max_mse, max_regret, min_reward
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from nldg.lr import MaggingLR
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_GRID = {
    "n_estimators": [60],
    "max_depth": [10, 15, 30],
    "min_samples_leaf": [5, 15, 25],
    "max_features": ["sqrt", "log2", 1.0],
}
# PARAMS_GRID_XGB = {
#     "n_estimators": [25, 50, 100],
#     "max_depth": [3, 6],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "subsample": [0.8, 1.0],
#     "colsample_bytree": [0.8, 1.0],
# }
PARAMS_GRID_XGB = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
PARAMS_GRID_RIDGE = {"alpha": [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
SCALE = 1e8
SEED = 42

# Set up root logger (only once)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_model(model_name, params=None):
    """
    Returns a model instance based on the model name and parameters.

    Args:
        model_name (str): The name of the model to instantiate.
        params (dict): Parameters to initialize the model.
    """
    if params is None:
        params = {}
    if model_name == "rf":
        return RandomForest(**params)
    elif model_name == "xgb":
        return XGBRegressor(**params)
    elif model_name == "lr":
        return LinearRegression()
    elif model_name == "ridge":
        return Ridge(**params)
    elif model_name == "maximin":
        return MaggingLR()
    else:
        raise NotImplementedError(f"Model `{model_name}` not implemented.")


def get_default_params(model_name, n_jobs=20):
    """
    Returns default parameters for the specified model.

    Args:
        model_name (str): The name of the model.
        n_jobs (int): The number of CPU cores to use.
    """
    params = {}
    if model_name == "rf":
        params = {
            "forest_type": "Regression",
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 30,
            "max_features": 1.0,
            "seed": SEED,
            "n_jobs": n_jobs,
        }
    elif model_name == "xgb":
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
    elif model_name == "ridge":
        params = {"alpha": 0.1}
    return params


def modify_predictions(
    model,
    train_ids_int,
    risk,
    group,
    sols_erm=None,
    sols_erm_trees=None,
    n_jobs=20,
    verbose=True,
):
    """
    Try modify_predictions_trees with several solvers, then fall back to
    opt_method='extragradient'. Returns True if successful, False otherwise.
    """
    kwargs = {"method": risk, "n_jobs": n_jobs}
    if risk == "regret":
        kwargs["sols_erm"] = sols_erm
        kwargs["sols_erm_trees"] = sols_erm_trees

    solvers = [None, "ECOS", "SCS"]

    for solver in solvers:
        try:
            if verbose:
                if solver is None:
                    logging.info(
                        f"* Trying default solver for group {group}..."
                    )
                else:
                    logging.info(
                        f"* Trying solver {solver} for group {group}..."
                    )
            model.modify_predictions_trees(
                train_ids_int, **kwargs, solver=solver
            )
            return True
        except Exception:
            if verbose:
                if solver is None:
                    logging.warning(
                        f"* Default solver failed for group {group}."
                    )
                else:
                    logging.warning(
                        f"* Solver {solver} failed for group {group}."
                    )

    if verbose:
        logging.warning(
            f"* Fallback [{group}]: all solvers failed. "
            "Retrying with opt_method='extragradient'."
        )
    try:
        model.modify_predictions_trees(
            train_ids_int, **kwargs, opt_method="extragradient"
        )
        return True
    except Exception:
        if verbose:
            logging.error(
                f"* SKIPPING {group}: Error in modify_predictions_trees after all fallbacks"
            )
        return False


def iter_grid(grid):
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def score_fold(
    xtrain,
    ytrain,
    train_ids_int,
    group,
    params_candidate,
    tr_idx,
    va_idx,
    model_name,
    method,
    risk,
    n_jobs,
    verbose=True,
):
    # split
    train_ids_int = np.asarray(train_ids_int)
    X_tr, y_tr = xtrain[tr_idx], ytrain[tr_idx].copy()
    X_va, y_va = xtrain[va_idx], ytrain[va_idx].copy()

    tr_ids = train_ids_int[tr_idx]
    va_ids = train_ids_int[va_idx]

    # scale
    y_tr_scaled = y_tr * SCALE

    # fit candidate
    model = get_model(model_name, params=params_candidate)
    model.fit(X_tr, y_tr_scaled)

    if model_name == "rf" and method == "maxrm":
        if risk == "regret":
            sols_erm_va = np.zeros(len(va_ids))
            sols_erm_tr = np.zeros(len(tr_ids))
            sols_erm_trees_tr = np.zeros(
                (params_candidate["n_estimators"], len(tr_ids))
            )
            for env in np.unique(tr_ids):
                mask = tr_ids == env
                xtrain_env = X_tr[mask]
                ytrain_env = y_tr_scaled[mask]
                rf_env = get_model("rf", params=params_candidate)
                rf_env.fit(xtrain_env, ytrain_env)
                fitted_env = rf_env.predict(xtrain_env)
                sols_erm_tr[mask] = fitted_env
                for i in range(params_candidate["n_estimators"]):
                    fitted_env_tree = rf_env.trees[i].predict(xtrain_env)
                    sols_erm_trees_tr[i, mask] = fitted_env_tree
                mask_va = va_ids == env
                sols_erm_va[mask_va] = rf_env.predict(X_va[mask_va]) / SCALE

        success = modify_predictions(
            model=model,
            train_ids_int=tr_ids,
            risk=risk,
            group=group,
            sols_erm=sols_erm_tr if risk == "regret" else None,
            sols_erm_trees=sols_erm_trees_tr if risk == "regret" else None,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if not success:
            return np.nan

    # validation
    y_pred_va = model.predict(X_va) / SCALE
    if method == "erm":
        return mean_squared_error(y_va, y_pred_va)
    if method == "maxrm":
        if risk == "mse":
            return max_mse(y_va, y_pred_va, va_ids)
        elif risk == "reward":
            return -min_reward(y_va, y_pred_va, va_ids)
        else:
            return max_regret(y_va, y_pred_va, sols_erm_va, va_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(BASE_DIR, "..", "data_cleaned"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["daily-50-2017", "daily-US-2021"],
        default="daily-50-2017",
        help="Data aggregation level",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=[
            "in-sites-grouped" "logo",
            "loso",
            "l5so",
        ],
        default="loso",
        help="Experiment setting",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["rf", "lr", "xgb", "ridge", "maximin"],
        default="rf",
        help="Model to use for the experiment",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="erm",
        choices=["erm", "maxrm"],
        help="Minimize the risk or minimize the maximum risk? (default: 'erm')."
        "Must be one of 'erm', 'maxrm'.",
    )
    parser.add_argument(
        "--risk",
        type=str,
        default="mse",
        choices=["mse", "reward", "regret"],
        help="Risk definition (default: 'mse')."
        "Must be one of 'mse', 'reward', 'regret'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for L5SO and in-sites-grouped strategies (default: 42).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation for the experiment",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=20,
        help="Number of jobs (default: 20).",
    )

    args = parser.parse_args()
    path = args.path
    agg = args.agg
    setting = args.setting
    model_name = args.model_name
    method = args.method
    risk = args.risk
    seed = args.seed
    cv = args.cv
    n_jobs = args.n_jobs

    if model_name == "rf":
        exp_name = f"{agg}_{setting}_{model_name}_{method}_{risk}"
    else:
        exp_name = f"{agg}_{setting}_{model_name}"

    # Get model parameters
    params = get_default_params(model_name, n_jobs)

    # Load data
    data_path = os.path.join(path, agg + ".csv")
    logging.info("Loading data...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

    # Set-up groups
    groups = generate_fold_info(df, setting, seed=seed)
    results = []

    # Run experiment
    for group_id, group in enumerate(groups):
        logging.info(f"Running group: {group}...")
        if cv:
            (
                xtrain,
                ytrain,
                xtest,
                ytest,
                train_ids,
                test_ids,
                cv_folds,
            ) = get_fold_df(
                df, setting, group, cv=cv, remove_missing=True, seed=seed
            )
        else:
            xtrain, ytrain, xtest, ytest, train_ids, test_ids = get_fold_df(
                df, setting, group, cv=cv, remove_missing=True, seed=seed
            )
        if xtrain is None:
            continue

        train_ids_int = train_ids.astype("category").cat.codes
        test_ids_int = test_ids.astype("category").cat.codes

        # ----------------
        # Cross-validation
        # -----------------
        if model_name in ["rf", "xgb", "ridge"] and cv:
            best_score = np.inf
            best_params = params
            if model_name == "rf":
                params_grid = PARAMS_GRID
            elif model_name == "ridge":
                params_grid = PARAMS_GRID_RIDGE
            else:
                params_grid = PARAMS_GRID_XGB
            for params_candidate in tqdm(iter_grid(params_grid), leave=False):
                if model_name == "rf":
                    params_candidate["forest_type"] = "Regression"
                    params_candidate["seed"] = SEED
                    params_candidate["n_jobs"] = n_jobs
                elif model_name == "xgb":
                    params_candidate["objective"] = "reg:squarederror"
                    params_candidate["random_state"] = SEED
                    params_candidate["verbosity"] = 0
                    params_candidate["n_jobs"] = n_jobs
                fold_scores = []
                for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds, start=1):
                    score = score_fold(
                        xtrain,
                        ytrain,
                        train_ids_int,
                        f"{group}-cv{fold_idx}",
                        params_candidate,
                        tr_idx,
                        va_idx,
                        model_name,
                        method,
                        risk,
                        n_jobs,
                        verbose=False,
                    )
                    fold_scores.append(score)
                if np.all(np.isnan(fold_scores)):
                    logging.warning(
                        f"* All CV folds failed for params {params_candidate}"
                    )
                    continue
                avg_score = np.nanmean(fold_scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = dict(params_candidate)
            params = best_params
            logging.info(f"* Params via CV: {params}")

        # -----------------------------------------
        # Fit on full training and evaluate on test
        # -----------------------------------------
        ytrain_scaled = ytrain * SCALE

        # Get model
        model = get_model(model_name, params=params)
        if model_name == "maximin":
            model.fit(xtrain, ytrain_scaled, train_ids_int)
        else:
            model.fit(xtrain, ytrain_scaled)

        # Just to compute the maximum regret across training environments
        if model_name == "rf":
            sols_erm = np.zeros(len(train_ids_int))
            sols_erm_trees = np.zeros(
                (params["n_estimators"], len(train_ids_int))
            )
            for env in np.unique(train_ids_int):
                mask = train_ids_int == env
                xtrain_env = xtrain[mask]
                ytrain_env = ytrain_scaled[mask]
                rf_env = get_model("rf", params=params)
                rf_env.fit(xtrain_env, ytrain_env)
                fitted_env = rf_env.predict(xtrain_env)
                sols_erm[mask] = fitted_env
                for i in range(params["n_estimators"]):
                    fitted_env_tree = rf_env.trees[i].predict(xtrain_env)
                    sols_erm_trees[i, mask] = fitted_env_tree

        if model_name == "rf" and method == "maxrm":
            success = modify_predictions(
                model=model,
                train_ids_int=train_ids_int,
                risk=risk,
                group=group,
                sols_erm=sols_erm if risk == "regret" else None,
                sols_erm_trees=sols_erm_trees if risk == "regret" else None,
                n_jobs=n_jobs,
            )
            if not success:
                continue

        # Evaluate model
        ypred = model.predict(xtest)
        ypred /= SCALE
        if setting == "loso":
            res = evaluate_fold(ytest, ypred, verbose=True, digits=3)
            res["group"] = group
        else:
            max_mse_test = max_mse(ytest, ypred, test_ids_int)
            res = {
                "max_mse_test": max_mse_test,
                "max_rmse_test": np.sqrt(max_mse_test),
                "group": group_id,
            }
            print(max_mse_test)
        if model_name == "rf":
            yfitted = model.predict(xtrain)
            yfitted /= SCALE
            sols_erm /= SCALE
            res["max_mse_train"] = max_mse(ytrain, yfitted, train_ids_int)
            res["max_nrw_train"] = -min_reward(ytrain, yfitted, train_ids_int)
            res["max_reg_train"] = max_regret(
                ytrain, yfitted, sols_erm, train_ids_int
            )
        results.append(res)

    # Save results
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(BASE_DIR, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{exp_name}.csv")
    logging.info(f"Saving results to {path}...")
    results_df.to_csv(path, index=False)
