# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import argparse
import numpy as np
import pandas as pd
from adaXT.random_forest import RandomForest
from nldg.additional.data import DataContainer
from nldg.additional.drol import DRoL
from nldg.utils import max_mse, min_reward, max_regret
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

N_SIM = 10
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
SEED = 42
COLORS = {
    "RF": "#5790FC",
    "MaxRM-RF": "#F89C20",
    "DRoL": "#964A8B",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_drol")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_maxrisk_vs_nenvs(
    results: dict[int, dict[str, float]],
    risk_label: str = "mse",
    change_X_distr: bool = False,
    unbalanced_envs: bool = False,
) -> None:
    methods = []
    for L in results:
        methods.extend(results[L].keys())
    methods = list(dict.fromkeys(methods))  # preserve order

    L_vals = sorted(results.keys())

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for m in methods:
        base = m.split("(")[0]
        if base not in COLORS.keys():
            continue
        color = COLORS.get(base, "#000000")

        xs, means, lowers, uppers = [], [], [], []
        for L in L_vals:
            vals = np.asarray(results[L].get(m, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                xs.append(L)
                means.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue
            n = vals.size
            mu = vals.mean()
            stderr = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            width = 1.96 * stderr

            xs.append(L)
            means.append(mu)
            lowers.append(mu - width)
            uppers.append(mu + width)

        ax.plot(
            xs,
            means,
            label=m,
            color=color,
            marker="o",
            linestyle="-",
            markeredgecolor="white",
        )
        ax.fill_between(xs, lowers, uppers, color=color, alpha=0.25)

    ax.set_xlabel("Number of test environments")
    if risk_label == "mse":
        ax.set_ylabel("Maximum MSE across test environments")
    elif risk_label == "nrw":
        ax.set_ylabel("Maximum Negative Reward across test environments")
    else:
        ax.set_ylabel("Maximum Regret across test environments")

    ax.set_xticks(L_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"comparison_drol_{risk_label}_changeXdistr{str(change_X_distr)}_unbalanced{str(unbalanced_envs)}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--change_X_distr",
        action="store_true",
        help="Whether the target covariate distribution differs from the training ones (default: False).",
    )
    parser.add_argument(
        "--unbalanced_envs",
        action="store_true",
        help="Whether the environments should be unbalanced (default: False).",
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
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs (default: 1).",
    )
    args = parser.parse_args()
    change_X_distr = args.change_X_distr
    unbalanced_envs = args.unbalanced_envs
    risk = args.risk
    n_jobs = args.n_jobs

    if risk == "mse":
        risk_label = "mse"
    elif risk == "reward":
        risk_label = "nrw"
    else:
        risk_label = "reg"

    Ls = [3, 4, 5, 6, 7, 8, 9, 10]  # number of environments
    results = {L: {} for L in Ls}
    for L in tqdm(Ls):
        if not unbalanced_envs:
            data = DataContainer(
                n=2000, N=20000, change_X_distr=change_X_distr, risk=risk
            )
        else:
            data = DataContainer(
                n=1000,
                N=20000,
                change_X_distr=change_X_distr,
                risk=risk,
                unbalanced_envs=unbalanced_envs,
            )
        data.generate_funcs_list(L=L, seed=SEED)
        max_risks = np.zeros((N_SIM, 3))  # RF, MaxRM-RF, DRoL
        for sim in tqdm(range(N_SIM), leave=False):
            data.generate_data(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)
            Etr = np.concatenate(data.E_sources_list)
            Xte = np.vstack(data.X_target_list)
            Yte = np.concatenate(data.Y_target_potential_list)
            Ete = np.concatenate(data.E_target_potential_list)

            if risk == "regret":
                fitted_erm = np.zeros(len(Etr))
                fitted_erm_trees = np.zeros((N_ESTIMATORS, len(Etr)))
                pred_erm = np.zeros(len(Ete))
                for env in np.unique(Etr):
                    mask = Etr == env
                    Xtr_env = Xtr[mask]
                    Ytr_env = Ytr[mask]
                    rf_e = RandomForest(
                        "Regression",
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        seed=SEED,
                        n_jobs=n_jobs,
                    )
                    rf_e.fit(Xtr_env, Ytr_env)
                    fitted_erm[mask] = rf_e.predict(Xtr_env)
                    for i in range(N_ESTIMATORS):
                        fitted_erm_trees[i, mask] = rf_e.trees[i].predict(
                            Xtr_env
                        )
                    mask_te = Ete == env
                    pred_erm[mask_te] = rf_e.predict(Xte[mask_te])

            # RF
            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=n_jobs,
            )
            rf.fit(Xtr, Ytr)
            pred_rf = rf.predict(Xte)

            # MaxRM-RF
            solvers = ["ECOS", "SCS"]
            success = False
            kwargs = {"n_jobs": n_jobs}
            if risk == "regret":
                kwargs["sols_erm"] = fitted_erm
                kwargs["sols_erm_trees"] = fitted_erm_trees
            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr,
                        method=risk,
                        **kwargs,
                        solver=solver,
                    )
                    success = True
                    break
                except Exception as e_try:
                    pass
            if not success:
                rf.modify_predictions_trees(
                    Etr,
                    method=risk,
                    **kwargs,
                    opt_method="extragradient",
                )
            pred_maxrmrf = rf.predict(Xte)

            # DRoL
            params = {
                "forest_type": "Regression",
                "n_estimators": N_ESTIMATORS,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "seed": SEED,
                "n_jobs": n_jobs,
            }
            sigma2 = np.ones(L)  # the noise term is N(0,1)
            drol = DRoL(data, params, method=risk, sigma2=sigma2, seed=SEED)
            try:
                drol.fit(density_learner="logistic")
                _pred_drol, weights = drol.predict(
                    bias_correct=True, priors=None
                )
                pred_drol = np.tile(_pred_drol, L)
            except Exception as e:
                pred_drol = None

            # Evaluate the maximum risk
            if risk == "mse":
                max_risks[sim, 0] = max_mse(Yte, pred_rf, Ete)
                max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf, Ete)
                if pred_drol is None:
                    max_risks[sim, 2] = np.nan
                else:
                    max_risks[sim, 2] = max_mse(Yte, pred_drol, Ete)
            elif risk == "reward":
                max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
                max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
                if pred_drol is None:
                    max_risks[sim, 2] = np.nan
                else:
                    max_risks[sim, 2] = -min_reward(Yte, pred_drol, Ete)
            else:
                max_risks[sim, 0] = max_regret(Yte, pred_rf, pred_erm, Ete)
                max_risks[sim, 1] = max_regret(
                    Yte, pred_maxrmrf, pred_erm, Ete
                )
                if pred_drol is None:
                    max_risks[sim, 2] = np.nan
                else:
                    max_risks[sim, 2] = max_regret(
                        Yte, pred_drol, pred_erm, Ete
                    )

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"DRoL({risk_label})"] = max_risks[:, 2].tolist()

    rows = []
    for L, methods in results.items():
        for method, vals in methods.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            n = arr.size
            if n == 0:
                continue
            mu = arr.mean()
            stderr = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            width = 1.96 * stderr
            rows.append(
                {
                    "L": L,
                    "method": method,
                    "n": n,
                    "mean": mu,
                    "lower": mu - width,
                    "upper": mu + width,
                }
            )
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(
        os.path.join(
            OUT_DIR,
            f"comparison_drol_{risk_label}_changeXdistr{str(change_X_distr)}_unbalanced{str(unbalanced_envs)}.csv",
        ),
        index=False,
    )

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
        change_X_distr=change_X_distr,
        unbalanced_envs=unbalanced_envs,
    )
