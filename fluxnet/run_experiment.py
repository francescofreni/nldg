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
from pygam import LinearGAM, s, l
from nldg.gam import MaxRMLinearGAM
from functools import reduce
from operator import add

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    elif model_name == "gam":
        return LinearGAM(**params)
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
        # default RF parameters in sklearn
        params = {
            "forest_type": "Regression",
            "n_estimators": 100,
            "seed": SEED,
            "n_jobs": n_jobs,
            "min_samples_leaf": 30,
        }
        # params = {
        #     "forest_type": "Regression",
        #     "n_estimators": 20,
        #     "max_depth": 8,
        #     "min_samples_leaf": 30,
        #     "max_features": 1.0,
        #     "seed": SEED,
        #     "n_jobs": n_jobs,
        # }
    elif model_name == "xgb":
        # default XGB parameters
        params = {
            "objective": "reg:squarederror",
            "random_state": SEED,
            "n_jobs": n_jobs,
        }
        # params = {
        #     "objective": "reg:squarederror",
        #     "n_estimators": 100,
        #     "max_depth": 5,
        #     "learning_rate": 0.1,
        #     "subsample": 1.0,
        #     "colsample_bytree": 1.0,
        #     "random_state": SEED,
        #     "n_jobs": n_jobs,
        #     "verbosity": 0,
        # }
    elif model_name == "ridge":
        params = {"alpha": 0.1}
    elif model_name == "gam":
        params = {"fit_intercept": True}
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
    ret_ind=False, # Whether to return also the MSE for each environment.
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
            return max_mse(y_va, y_pred_va, va_ids, ret_ind=ret_ind)
        elif risk == "reward":
            return -min_reward(y_va, y_pred_va, va_ids, ret_ind=ret_ind)
        else:
            return max_regret(y_va, y_pred_va, sols_erm_va, va_ids, ret_ind=ret_ind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(BASE_DIR, "data_cleaned"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["daily-50-2017", "daily-US-2021", "daily"],
        default="daily-50-2017",
        help="Data aggregation level",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=[
            "in-sites-grouped",
            "logo",
            "loso",
            "l5so",
            "insite"
        ],
        default="loso",
        help="Experiment setting",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["rf", "lr", "xgb", "ridge", "maximin", "gam"],
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
        help="Seed for l5so and in-sites-grouped strategies (default: 42).",
    )
    parser.add_argument(
        "--fold_size",
        type=int,
        default=10,
        help="Fold size for the in-sites-grouped strategy, ignored if setting is l5so (default: 10).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=20,
        help="Number of jobs used for rf and xgb (default: 20).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="GPP",
        help="Target variable to predict (default: 'GPP').",
        choices=["GPP", "NEE"],
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional experiment name to save results under",
    )

    args = parser.parse_args()
    path = args.path
    agg = args.agg
    setting = args.setting
    model_name = args.model_name
    method = args.method
    risk = args.risk
    seed = args.seed
    fold_size = args.fold_size
    n_jobs = args.n_jobs
    target = args.target
    exp_name = args.exp_name
    cv = False

    # TODO: implement cv with MaxRM-AM
    if model_name == "gam" and method == "maxrm" and cv:
        raise ValueError("cv not implemented yet for MaxRM-AM.")
    
    if method == "erm" and risk != "mse":
        raise ValueError("`risk` must be 'mse' when `method` is 'erm'.")

    # Set up experiment name and file path
    if exp_name is not None:
        os.makedirs(os.path.join(BASE_DIR, "results", exp_name), exist_ok=True)
    exp_name = f"{exp_name}/" if exp_name is not None else ""
    exp_name += f"{agg}_{setting}_{target}_{model_name}_{method}_{risk}"
    # if model_name in ["rf", "gam"]:
    #     exp_name += f"{agg}_{setting}_{target}_{model_name}_{method}_{risk}"
    # else:
    #     exp_name += f"{agg}_{setting}_{target}_{model_name}"

    # Get model parameters
    params = get_default_params(model_name, n_jobs)

    # Load data
    data_path = os.path.join(path, agg + ".csv")
    logging.info("Loading data...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)
    df = df.dropna(subset=[target])

    # Set-up groups
    min_samples_per_env = 150 if setting == "insite" else None
    if setting == "in-sites-grouped":
        groups = generate_fold_info(
            df, setting, fold_size=fold_size, seed=seed
        )
    else:
        groups = generate_fold_info(df, setting, seed=seed)
    results = []

    # Run experiment
    for group_id, group in enumerate(groups):
        logging.info(f"Running group: {group}...")
        xtrain, ytrain, xtest, ytest, train_ids, test_ids = get_fold_df(
            df,
            setting,
            group,
            cv=cv,
            remove_missing=True,
            target=target,
            min_samples=min_samples_per_env
        )
        if xtrain is None:
            continue

        train_ids_int = train_ids.astype("category").cat.codes
        test_ids_int = test_ids.astype("category").cat.codes

        # -----------------------------------------
        # Fit on full training and evaluate on test
        # -----------------------------------------
        ytrain_scaled = ytrain * SCALE

        # compute the erm solution for gam
        if model_name == "gam":
            sols_erm = np.zeros(len(train_ids_int))
            for env in np.unique(train_ids_int):
                mask = train_ids_int == env
                xtrain_env = xtrain[mask]
                ytrain_env = ytrain_scaled[mask]
                terms_env = [l(0), l(1)] + [
                    s(j) for j in range(2, xtrain_env.shape[1])
                ]
                params["terms"] = reduce(add, terms_env)
                am_env = get_model(model_name, params=params)
                am_env.fit(xtrain_env, ytrain_env)
                fitted_env = am_env.predict(xtrain_env)
                sols_erm[mask] = fitted_env

        # Get model
        if model_name == "gam":
            terms = [l(0), l(1)] + [s(j) for j in range(2, xtrain.shape[1])]
            params["terms"] = reduce(add, terms)

        if model_name == "gam" and method == "maxrm":
            model = MaxRMLinearGAM(**params)
        else:
            model = get_model(model_name, params=params)

        if model_name == "maximin":
            model.fit(xtrain, ytrain_scaled, train_ids_int)
        elif model_name == "gam" and method == "maxrm":
            model.fit(
                xtrain,
                ytrain_scaled,
                train_ids_int,
                risk=risk,
                sols_erm=sols_erm,
            )
        else:
            if model_name == "gam" and method == "erm" and cv:
                model.gridsearch(xtrain, ytrain_scaled)
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
                logging.error(f"SKIPPING {group}: Error in modify_predictions_trees")
                continue

        # Evaluate model
        ypred = model.predict(xtest)
        ypred /= SCALE
        if setting == "loso":
            res = evaluate_fold(ytest, ypred, verbose=True, digits=3)
            res["group"] = group
        else:
            mse_all, max_mse_test = max_mse(ytest, ypred, test_ids_int, ret_ind=True)
            res = {
                "max_mse_test": max_mse_test,
                "max_rmse_test": np.sqrt(max_mse_test),
                "avg_mse_test": np.mean(mse_all),
                "avg_rmse_test": np.mean(np.sqrt(mse_all)),
                "group": group_id,
            }
            print(max_mse_test)
        if model_name in ["rf", "gam"]:
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
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{exp_name}.csv")
    logging.info(f"Saving results to {path}...")
    results_df.to_csv(path, index=False)
