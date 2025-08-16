# Code modified from https://github.com/anyafries/fluxnet_bench.git
import argparse
import logging
import os
import numpy as np
import pandas as pd
from adaXT.random_forest import RandomForest
from dataloader import generate_fold_info, get_fold_df
from eval import evaluate_fold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up root logger (only once)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_model(model_name, params={}):
    """
    Returns a model instance based on the model name and parameters.

    Args:
        model_name (str): The name of the model to instantiate.
        params (dict): Parameters to initialize the model.
    """
    if model_name == "rf":
        return RandomForest(**params)
    else:
        raise NotImplementedError(f"Model `{model_name}` not implemented.")


def get_default_params(model_name):
    """
    Returns default parameters for the specified model.

    Args:
        model_name (str): The name of the model.
    """
    params = {}
    if model_name == "rf":
        params = {
            "forest_type": "Regression",
            "n_estimators": 20,
            "min_samples_leaf": 50,
            "max_depth": 8,
            "seed": 42,
            "n_jobs": 20,
        }
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(BASE_DIR, "data_cleaned"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing results"
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["seasonal", "daily", "raw", "daily10", "daily30", "daily50"],
        default="daily10",
        help="Data aggregation level",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["insite", "insite-random", "logo", "loso"],
        default="loso",
        help="Experiment setting",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for the experiment"
    )
    parser.add_argument(
        "--stop", type=int, default=None, help="Stop index for the experiment"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["rf"],
        default="rf",
        help="Model to use for the experiment",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom name for the experiment",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameter file for the model",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation for the experiment",
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
        "--solver",
        type=str,
        default=None,
        choices=["CLARABEL", "SCS", "ECOS"],
        help="Interior-point solver to use for MaxRM Random Forest (default: None)."
        "Must be one of None, 'CLARABEL', 'SCS', 'ECOS'.",
    )

    args = parser.parse_args()
    path = args.path
    override = args.override
    agg = args.agg
    setting = args.setting
    start = args.start
    stop = args.stop
    model_name = args.model_name
    exp_name = args.experiment_name
    params = args.params
    cv = args.cv
    method = args.method
    risk = args.risk
    solver = args.solver

    if exp_name is None:
        if model_name == "rf":
            if method == "erm":
                exp_name = f"{agg}_{setting}_RF_start{start}_stop{stop}"
            else:
                if risk == "mse":
                    exp_name = f"{agg}_{setting}_MaxRM-RF(posthoc-mse)_start{start}_stop{stop}"
                elif risk == "reward":
                    exp_name = f"{agg}_{setting}_MaxRM-RF(posthoc-nrw)_start{start}_stop{stop}"
                else:
                    exp_name = f"{agg}_{setting}_MaxRM-RF(posthoc-reg)_start{start}_stop{stop}"
        else:
            exp_name = f"{agg}_{setting}_{model_name}_start{start}_stop{stop}"

    # Get model parameters
    if params is not None and os.path.exists(params):
        params = pd.read_csv(params)
        params = params.to_dict(orient="records")[0]
    else:
        params = get_default_params(model_name)

    # Load data
    data_path = os.path.join(path, agg + ".csv")
    logging.info("Loading data...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

    # Set-up groups
    groups = generate_fold_info(df, setting, start, stop)
    results = []

    # Run experiment
    for group in groups:
        logging.info(f"Running group: {group}...")
        xtrain, ytrain, xtest, ytest, train_ids = get_fold_df(
            df, setting, group, cv=cv, remove_missing=True
        )
        if xtrain is None:
            continue

        if method == "maxrm":
            train_ids_int = train_ids.astype("category").cat.codes

        if model_name == "rf":
            ytrain *= 1e8

        # Get model
        model = get_model(model_name, params=params)
        model.fit(xtrain, ytrain)
        if model_name == "rf" and method == "maxrm":
            try:
                if risk == "mse":
                    model.modify_predictions_trees(
                        train_ids_int,
                        solver=solver,
                        n_jobs=params["n_jobs"],
                    )
                elif risk == "reward":
                    model.modify_predictions_trees(
                        train_ids_int,
                        method="reward",
                        solver=solver,
                        n_jobs=params["n_jobs"],
                    )
                else:
                    sols_erm = np.zeros(len(train_ids_int))
                    sols_erm_trees = np.zeros(
                        (params["n_estimators"], len(train_ids_int))
                    )
                    for env in np.unique(train_ids_int):
                        mask = train_ids_int == env
                        xtrain_env = xtrain[mask]
                        ytrain_env = ytrain[mask]
                        rf_env = RandomForest(**params)
                        rf_env.fit(xtrain_env, ytrain_env)
                        fitted_env = rf_env.predict(xtrain_env)
                        sols_erm[mask] = rf_env.predict(xtrain_env)
                        for i in range(params["n_estimators"]):
                            fitted_env_tree = rf_env.trees[i].predict(
                                xtrain_env
                            )
                            sols_erm_trees[i, mask] = fitted_env_tree
                    model.modify_predictions_trees(
                        train_ids_int,
                        method="regret",
                        sols_erm=sols_erm,
                        sols_erm_trees=sols_erm_trees,
                        solver=solver,
                        n_jobs=params["n_jobs"],
                    )
            except Exception as e:
                logging.error(
                    f"* SKIPPING {group}: Error in modify_predictions_trees"
                )
                continue

        # Evaluate model
        ypred = model.predict(xtest)
        if model_name == "rf":
            ypred /= 1e8
        res = evaluate_fold(ytest, ypred, verbose=True, digits=3)
        res["group"] = group
        results.append(res)

    # Save results
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{exp_name}.csv")
    logging.info(f"Saving results to {path}...")
    results_df.to_csv(path, index=False)
