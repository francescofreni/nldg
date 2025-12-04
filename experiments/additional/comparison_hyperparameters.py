import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from adaXT.random_forest import RandomForest
from nldg.utils import max_mse, max_regret, min_reward
from nldg.additional.data_GP import DataContainer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from experiments.utils import plot_max_risk_vs_hyperparam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_hyperparameters")
os.makedirs(OUT_DIR, exist_ok=True)

N_SIM = 100
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF_DEFAULT = 30
SEED = 42
N_ENVS = 5
NUM_COVARIATES = 5
BETA_LOW = 0.5
BETA_HIGH = 2.5
NAME_RF = "MaxRM-RF"


# hyperparameter grids
GRID_MTRY = np.arange(1, NUM_COVARIATES + 1)
GRID_MIN_SAMPLES_LEAF = np.array([1, 3, 5, 10, 15, 20, 25, 30, 35, 40])
GRID_MAX_DEPTH = np.array([3, 4, 5, 6, 7, 8])


def run_experiment(args, hyperparam_name, grid):
    risk_name = args.risk
    change_X_distr = args.change_X_distr
    n_jobs = args.n_jobs
    risk_suffix = args.risk_suffix

    logger.info(
        f"Starting experiment for {hyperparam_name} with risk={risk_name}"
    )

    results = []

    for sim in tqdm(range(N_SIM), desc=f"Simulations ({hyperparam_name})"):
        data = DataContainer(
            n=2000,
            N=2000,
            L=N_ENVS,
            d=NUM_COVARIATES,
            change_X_distr=change_X_distr,
            risk=risk_name,
            beta_low=BETA_LOW,
            beta_high=BETA_HIGH,
        )
        data.generate_dataset(seed=sim)

        Xtr = np.vstack(data.X_sources_list)
        Ytr = np.concatenate(data.Y_sources_list)
        Etr = np.concatenate(data.E_sources_list)
        Xte = np.vstack(data.X_target_list)
        Yte = np.concatenate(data.Y_target_potential_list)
        Ete = np.concatenate(data.E_target_potential_list)

        for val in grid:
            rf_params = {
                "n_estimators": N_ESTIMATORS,
                "min_samples_leaf": MIN_SAMPLES_LEAF_DEFAULT,
                "seed": SEED,
                "n_jobs": n_jobs,
                "forest_type": "Regression",
            }

            # ERM baselines for regret
            fitted_erm = None
            fitted_erm_trees = None
            pred_erm = None

            if risk_name == "regret":
                fitted_erm = np.zeros(len(Etr))
                fitted_erm_trees = np.zeros((N_ESTIMATORS, len(Etr)))
                pred_erm = np.zeros(len(Ete))

                for env in np.unique(Etr):
                    mask = Etr == env
                    Xtr_env = Xtr[mask]
                    Ytr_env = Ytr[mask]

                    # RF for specific environment
                    rf_e = RandomForest(**rf_params)
                    rf_e.fit(Xtr_env, Ytr_env)

                    fitted_erm[mask] = rf_e.predict(Xtr_env)
                    for k in range(N_ESTIMATORS):
                        fitted_erm_trees[k, mask] = rf_e.trees[k].predict(
                            Xtr_env
                        )

                    mask_te = Ete == env
                    pred_erm[mask_te] = rf_e.predict(Xte[mask_te])

            if hyperparam_name == "mtry":
                rf_params["max_features"] = int(val)
            elif hyperparam_name == "min_samples_leaf":
                rf_params["min_samples_leaf"] = int(val)
            elif hyperparam_name == "max_depth":
                rf_params["max_depth"] = int(val)

            # train RF
            rf = RandomForest(**rf_params)
            rf.fit(Xtr, Ytr)

            # predict RF
            pred_rf = rf.predict(Xte)

            # evaluate RF
            risk_val_rf = 0.0
            if risk_name == "mse":
                risk_val_rf = max_mse(Yte, pred_rf, Ete)
            elif risk_name == "reward":
                risk_val_rf = -min_reward(Yte, pred_rf, Ete)
            elif risk_name == "regret":
                risk_val_rf = max_regret(Yte, pred_rf, pred_erm, Ete)

            results.append(
                {
                    "method": "RF",
                    hyperparam_name: val,
                    "risk": risk_val_rf,
                    "sim": sim,
                }
            )

            # MaxRM-RF
            solvers = ["ECOS", "CLARABEL", "SCS"]
            success = False
            kwargs = {"n_jobs": n_jobs}
            if risk_name == "regret":
                kwargs["sols_erm"] = fitted_erm
                kwargs["sols_erm_trees"] = fitted_erm_trees

            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr,
                        method=risk_name,
                        **kwargs,
                        solver=solver,
                    )
                    success = True
                    break
                except Exception:
                    pass

            if not success:
                try:
                    rf.modify_predictions_trees(
                        Etr,
                        method=risk_name,
                        **kwargs,
                        opt_method="extragradient",
                    )
                except Exception as e:
                    logger.warning(f"MaxRM optimization failed: {e}")
                    pass

            pred_maxrm = rf.predict(Xte)

            # evaluate MaxRM-RF
            risk_val_maxrm = 0.0
            if risk_name == "mse":
                risk_val_maxrm = max_mse(Yte, pred_maxrm, Ete)
            elif risk_name == "reward":
                risk_val_maxrm = -min_reward(Yte, pred_maxrm, Ete)
            elif risk_name == "regret":
                risk_val_maxrm = max_regret(Yte, pred_maxrm, pred_erm, Ete)

            results.append(
                {
                    "method": f"{NAME_RF}({risk_suffix})",
                    hyperparam_name: val,
                    "risk": risk_val_maxrm,
                    "sim": sim,
                }
            )

    df_res = pd.DataFrame(results)

    plot_name = f"max_{risk_suffix}_{hyperparam_name}_changeX_{change_X_distr}"

    plot_max_risk_vs_hyperparam(
        df_res,
        hyperparam=hyperparam_name,
        saveplot=True,
        nameplot=plot_name,
        out_dir=OUT_DIR,
        suffix=risk_suffix,
    )

    csv_path = os.path.join(OUT_DIR, f"{plot_name}.csv")
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk", type=str, default="mse", choices=["mse", "reward", "regret"]
    )
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--change_X_distr", action="store_true")
    args = parser.parse_args()

    if args.risk == "mse":
        args.risk_suffix = "mse"
    elif args.risk == "reward":
        args.risk_suffix = "nrw"
    else:
        args.risk_suffix = "reg"

    run_experiment(args=args, hyperparam_name="mtry", grid=GRID_MTRY)
    run_experiment(
        args=args,
        hyperparam_name="min_samples_leaf",
        grid=GRID_MIN_SAMPLES_LEAF,
    )
    run_experiment(args=args, hyperparam_name="max_depth", grid=GRID_MAX_DEPTH)
