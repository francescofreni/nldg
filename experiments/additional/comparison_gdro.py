# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import argparse
import numpy as np
import pandas as pd
from adaXT.random_forest import RandomForest
from nldg.additional.data import DataContainer
from nldg.additional.gdro import GroupDRO
from nldg.utils import max_mse, min_reward
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
    "GroupDRO-NN": "#964A8B",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_gdro")
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
            vals = np.asarray(results[L][m], dtype=float)
            if vals.size == 0:
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
    else:
        ax.set_ylabel("Maximum Negative Reward across test environments")

    ax.set_xticks(L_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"comparison_gdro_{risk_label}_changeXdistr{str(change_X_distr)}_unbalanced{str(unbalanced_envs)}.pdf",
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
        choices=["mse", "reward"],
        help="Risk definition (default: 'mse')."
        "Must be one of 'mse', 'reward'.",
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
    else:
        risk_label = "nrw"

    Ls = [3, 4, 5, 6, 7, 8, 9, 10]  # number of environments
    results = {L: {} for L in Ls}
    for L in tqdm(Ls):
        if not unbalanced_envs:
            data = DataContainer(
                n=2000, N=2000, change_X_distr=change_X_distr, risk=risk
            )
        else:
            data = DataContainer(
                n=1000,
                N=1000,
                change_X_distr=change_X_distr,
                risk=risk,
                unbalanced_envs=unbalanced_envs,
            )

        data.generate_funcs_list(L=L, seed=SEED)
        max_risks = np.zeros((N_SIM, 4))  # RF, MaxRM-RF, GroupDRO, Mean

        overall_vars = np.zeros(N_SIM)
        per_env_vars = np.zeros((N_SIM, L))

        for sim in tqdm(range(N_SIM), leave=False):
            data.generate_data(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)
            Etr = np.concatenate(data.E_sources_list)
            Xte = np.vstack(data.X_target_list)
            Yte = np.concatenate(data.Y_target_potential_list)
            Ete = np.concatenate(data.E_target_potential_list)

            overall_vars[sim] = np.var(Yte)
            for i, Y_target in enumerate(data.Y_target_potential_list):
                per_env_vars[sim, i] = np.var(Y_target)

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
            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr, method=risk, solver=solver, n_jobs=n_jobs
                    )
                    success = True
                    break
                except Exception as e_try:
                    pass
            if not success:
                rf.modify_predictions_trees(
                    Etr,
                    method=risk,
                    n_jobs=n_jobs,
                    opt_method="extragradient",
                )
            pred_maxrmrf = rf.predict(Xte)

            # GroupDRO-NN
            gdro = GroupDRO(
                data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk=risk
            )
            gdro.fit(epochs=500)
            pred_gdro = gdro.predict(Xte)

            # Mean
            pred_mean = np.full(len(Yte), np.mean(Ytr))

            # Evaluate the maximum risk
            if risk == "mse":
                max_risks[sim, 0] = max_mse(Yte, pred_rf, Ete)
                max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = max_mse(Yte, pred_gdro, Ete)
                max_risks[sim, 3] = max_mse(Yte, pred_mean, Ete)
            else:
                max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
                max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = -min_reward(Yte, pred_gdro, Ete)
                max_risks[sim, 3] = -min_reward(Yte, pred_mean, Ete)

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"GroupDRO-NN({risk_label})"] = max_risks[:, 2].tolist()
        results[L][f"Mean"] = max_risks[:, 3].tolist()

        print(
            f"Overall variance: {np.mean(overall_vars):.3f} +- {np.std(overall_vars):.3f}"
        )
        for i in range(L):
            print(
                f"Variance target {i}: {np.mean(per_env_vars[:, i]):.3f} +- {np.std(per_env_vars[:, i]):.3f}"
            )

    # print(results)

    rows = []
    for L, methods in results.items():
        for method, vals in methods.items():
            arr = np.asarray(vals, dtype=float)
            n = arr.size
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
            f"comparison_gdro_{risk_label}_changeXdistr{str(change_X_distr)}_unbalanced{str(unbalanced_envs)}.csv",
        ),
        index=False,
    )

    # rows = []
    # for L, methods in results.items():
    #     for method, value in methods.items():
    #         rows.append({"L": L, "method": method, "risk": value})
    # df = pd.DataFrame(rows)
    # df.to_csv(
    #     os.path.join(
    #         OUT_DIR, f"comparison_gdro_{risk_label}_{str(change_X_distr)}.csv"
    #     ),
    #     index=False,
    # )

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
        change_X_distr=change_X_distr,
        unbalanced_envs=unbalanced_envs,
    )
