# Code modified from https://github.com/anyafries/fluxnet_bench.git
import argparse
import cvxpy as cp
import logging
import os
import numpy as np
import pandas as pd

from scipy.optimize import linprog
from dataloader import generate_fold_info, get_fold_df, preprocess_distance

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


def in_hull(points, x, tol=1e-9):
    n_points = len(points)
    c = np.zeros(n_points)  # objective is irrelevant, feasibility problem
    A_eq = np.r_[points.T, np.ones((1, n_points))]
    b_eq = np.r_[x, 1]
    
    # enforce λ_i >= 0
    bounds = [(0, None)] * n_points
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.success and res.status == 0 and res.fun <= tol


def distance_to_hull(points, x):
    n, d = points.shape
    lambdas = cp.Variable(n)
    combo = points.T @ lambdas
    objective = cp.Minimize(cp.norm(combo - x, 2))
    constraints = [lambdas >= 0, cp.sum(lambdas) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return prob.value


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
        choices=[
            "seasonal",
            "daily",
            "raw",
            "daily-50-2017",
        ],
        default="daily-50-2017",
        help="Data aggregation level",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=[
            "insite",
            "insite-random",
            "logo",
            "loso",
            "l5so",
        ],
        default="loso",
        help="Experiment setting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for L5SO strategy (default: 42).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["in_hull", "dist_hull"],
        default="in_hull",
        help="Extrapolation metric to compute",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalize features before distance calculation",
    )
    parser.add_argument(
        "--include-gpp",
        action="store_true",
        help="Whether to include GPP in distance calculation",
    )

    args = parser.parse_args()
    path = args.path
    agg = args.agg
    setting = args.setting
    seed = args.seed
    metric = args.metric
    norm = args.norm
    include_gpp = args.include_gpp

    exp_name = f"{agg}_{setting}"
    if norm:
        exp_name += "_norm"
    if include_gpp:
        exp_name += "_gpp"

    # Load data
    data_path = os.path.join(path, agg + ".csv")
    logging.info("Loading data...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

    # Set-up groups
    groups = generate_fold_info(df, setting, seed=seed)
    results = []
    results_site_level = []

    # Run experiment
    for group_id, group in enumerate(groups):
        logging.info(f"Running group: {group}...")
        (
            xtrain,
            ytrain,
            xtest,
            ytest,
            train_ids,
            test_ids,
            xcols,
        ) = get_fold_df(
            df,
            setting,
            group,
            remove_missing=True,
        )
        if xtrain is None:
            continue

        train_ids_int = train_ids.astype("category").cat.codes
        test_ids_int = test_ids.astype("category").cat.codes

        # get the subset of xcols appropriate for distance calc
        ytrain = ytrain if include_gpp else None
        ytest = ytest if include_gpp else None
            
        dist_train, dist_test = preprocess_distance(
            xtrain, xtest, xcols, norm=norm,
            ytrain=ytrain, ytest=ytest)

        sites_test = list(np.unique(test_ids))
        if metric == "in_hull":
            num_in_hull = 0
            for site in sites_test:
                num_in_hull_site = 0
                mask_site = (test_ids == site).values
                xtest_site = dist_test[mask_site,:]
                for i in range(xtest_site.shape[0]):
                    row = xtest_site[i,:]
                    row_in_hull = in_hull(dist_train, row)
                    num_in_hull += row_in_hull
                    num_in_hull_site += row_in_hull
                    results_site_level.append({
                        "group": group_id,
                        "site": site,
                        "metric": "in_hull",
                        "value": row_in_hull,
                        "i": i
                    })
                perc_in_hull_site = num_in_hull_site / xtest_site.shape[0]
                results.append({
                    "group": group_id,
                    "site": site,
                    "metric": "in_hull",
                    "value": perc_in_hull_site
                })
                logging.info(f"Site {site}: {num_in_hull_site} / {xtest_site.shape[0]} = {perc_in_hull_site:.2f}")
            perc_in_hull = num_in_hull / dist_test.shape[0]
            results.append({
                "group": group_id,
                "site": "all",
                "metric": "in_hull",
                "value": perc_in_hull
            })
            logging.info(f"Total: {num_in_hull} / {dist_test.shape[0]} = {perc_in_hull:.2f}")

        elif metric == "dist_hull":
            dist = 0
            for site in sites_test:
                dist_site = 0
                mask_site = (test_ids == site).values
                xtest_site = dist_test[mask_site,:]
                for i in range(xtest_site.shape[0]):
                    row = xtest_site[i,:]
                    dist_i = distance_to_hull(dist_train, row)
                    dist += dist_i
                    dist_site += dist_i
                    results_site_level.append({
                        "group": group_id,
                        "site": site,
                        "metric": "in_hull",
                        "value": dist_i,
                        "i": i
                    })
                avg_dist_site = dist_site / xtest_site.shape[0]
                results.append({
                    "group": group_id,
                    "site": site,
                    "metric": "dist_hull",
                    "value": avg_dist_site
                })
                logging.info(f"Site {site}: avg distance to hull = {avg_dist_site:.4f}")
            avg_dist = dist / dist_test.shape[0]
            results.append({
                "group": group_id,
                "site": "all",
                "metric": "dist_hull",
                "value": avg_dist
            })
            logging.info(f"Total: avg distance to hull = {avg_dist:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(BASE_DIR, "results/extrapolation_measures")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{metric}_{exp_name}.csv")
    logging.info(f"Saving results to {path}...")
    results_df.to_csv(path, index=False)

    path2 = os.path.join(results_dir, f"{metric}_site_level_{exp_name}.csv")
    logging.info(f"Saving site-level results to {path2}...")
    results_df2 = pd.DataFrame(results_site_level)
    results_df2.to_csv(path2, index=False)
