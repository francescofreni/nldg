"""Simulation script to compare different options for indeterminate leaves"""

# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
import argparse
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP import DataContainer
from nldg.utils import min_reward, max_mse, max_regret
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.pyplot as plt


N_SIM = 2
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 30
SEED = 42
COLORS = [
    "#F89C20",
    "#5790FC",
]


NUM_COVARIATES = 5
BETA_LOW = 0.5
BETA_HIGH = 2.5


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_indeterminate_leaves")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_maxrisk_vs_nenvs(
    results: dict[int, dict[str, float]],
    change_X_distr: bool,
    ylim: tuple[float, float] | None = None,
    risk_eval: str = "mse",
    risk_label: str = "mse",
) -> None:
    """Plot (with 95% normal-approx CI) the max env risk versus number of envs.

    Parameters
    ----------
    results : dict[int, dict[str, list[float]]]
    change_X_distr : bool
        Whether target covariate distribution differs from training sources
        (affects filename).
    ylim : (float, float) | None
        Optional y-axis limits.
    risk_eval : str
        Risk used for evaluation
    risk_label : str
        Short label of optimized risk for embedding in output filenames.
    """
    fontsize = 16

    L_vals = sorted(results.keys())

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    methods = results[L_vals[0]].keys()

    for i, m in enumerate(methods):
        color = COLORS[i]

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

    ax.set_xlabel("Number of environments $K$", fontsize=fontsize)
    if risk_eval == "mse":
        ax.set_ylabel("Maximum MSE across environments", fontsize=fontsize)
    elif risk_eval == "reward":
        ax.set_ylabel(
            "Maximum negative reward\nacross environments", fontsize=fontsize
        )
    elif risk_eval == "regret":
        ax.set_ylabel("Maximum regret across environments", fontsize=fontsize)

    ax.set_xticks(L_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=True, fontsize=fontsize)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ymin, ymax = ax.get_ylim()
    start = np.ceil((ymin + 1e-9) / 0.1) * 0.1
    ticks = np.arange(start, ymax, 0.1)
    ticks = [t for t in ticks if ymin < t < ymax]
    ax.set_yticks(ticks)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"opt{risk_label}_eval{risk_eval}_changeXdistr{str(change_X_distr)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


def plot_indeterminate_prop_vs_nenvs(
    total_indeterminate_prop: np.ndarray,
    Ls: list[int],
    change_X_distr: bool,
    risk_eval: str = "mse",
    risk_label: str = "mse",
) -> None:
    """Plot (with 95% normal-approx CI) the proportion of indeterminate leaves versus number of envs."""
    fontsize = 16

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    color = "#5790FC"

    xs, means, lowers, uppers = [], [], [], []
    for i, L in enumerate(Ls):
        vals = total_indeterminate_prop[i, :] * 100
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
        label="Indeterminate leaves",
        color=color,
        marker="o",
        linestyle="-",
        markeredgecolor="white",
    )
    ax.fill_between(xs, lowers, uppers, color=color, alpha=0.25)

    ax.set_xlabel("Number of environments $K$", fontsize=fontsize)
    ax.set_ylabel("Proportion of indeterminate leaves", fontsize=fontsize)

    ax.set_xticks(Ls)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True, linewidth=0.2)
    # ax.legend(frameon=True, fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"indeterminate_prop_opt{risk_label}_eval{risk_eval}_changeXdistr{str(change_X_distr)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.pdf",
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
        default=5,
        help="Number of jobs (default: 1).",
    )
    args = parser.parse_args()
    change_X_distr = args.change_X_distr
    risk = args.risk
    n_jobs = args.n_jobs

    if risk == "mse":
        risk_label = "mse"
    elif risk == "reward":
        risk_label = "nrw"
    else:
        risk_label = "reg"

    # evaluate on the same risk as optimized
    risk_eval = risk

    Ls = [2, 3, 4, 5, 6, 7, 8]  # number of environments
    results = {L: {} for L in Ls}
    total_indeterminate_counts = np.zeros((len(Ls), N_SIM))
    total_indeterminate_prop = np.zeros((len(Ls), N_SIM))

    for L in tqdm(Ls):
        # max_risks columns: [MaxRM-RF, MaxRM-RF (revert)]
        max_risks = np.zeros((N_SIM, 2))

        for sim in tqdm(range(N_SIM), leave=False):
            # 1. data generation per simulation
            data = DataContainer(
                n=2000,
                N=2000,
                L=L,
                d=NUM_COVARIATES,
                change_X_distr=change_X_distr,
                risk=risk,
                beta_low=BETA_LOW,
                beta_high=BETA_HIGH,
            )

            # resample functions for each simulation
            data.generate_dataset(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)
            Etr = np.concatenate(data.E_sources_list)
            Xte = np.vstack(data.X_target_list)
            Yte = np.concatenate(data.Y_target_potential_list)
            Ete = np.concatenate(data.E_target_potential_list)

            fitted_erm = None
            fitted_erm_trees = None
            pred_erm = None

            # 2. Regret-specific per-env ERM baselines (only if risk == "regret")
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

            # 3. Fit pooled RF
            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=n_jobs,
            )
            rf.fit(Xtr, Ytr)
            # pred_rf = rf.predict(Xte)
            # ---------------------------------------------------------------

            # 4. Modify RF predictions to obtain MaxRM-RF
            solvers = ["ECOS", "SCS", "CLARABEL"]
            success = False
            kwargs = {"n_jobs": n_jobs}
            if risk == "regret":
                kwargs["sols_erm"] = fitted_erm
                kwargs["sols_erm_trees"] = fitted_erm_trees
            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr, method=risk, **kwargs, solver=solver
                    )
                    success = True
                    break
                except Exception:
                    pass
            if not success:
                print(
                    f"All solvers failed for MaxRM-RF with risk={risk} in sim {sim}, L={L}."
                )
                rf.modify_predictions_trees(
                    Etr,
                    method=risk,
                    **kwargs,
                    opt_method="extragradient",
                )

            indet_counts = rf.indeterminate_counts

            if indet_counts is None:
                total_indeterminate_counts[Ls.index(L), sim] = 0.0
            else:
                total_indeterminate_counts[Ls.index(L), sim] = np.sum(
                    np.array(rf.indeterminate_counts)
                )

            # total number of leaves
            total_leaves = np.sum([len(tree.leaf_nodes) for tree in rf.trees])

            total_indeterminate_prop[Ls.index(L), sim] = (
                total_indeterminate_counts[Ls.index(L), sim] / total_leaves
            )

            pred_maxrmrf = rf.predict(Xte)
            pred_maxrmrf_revert = rf.predict(Xte, revert_to_rf=True)

            # 5. Replace any infinite MaxRM-RF predictions (solver fallback safeguard).
            # sometimes convex solver fails
            # -> replace any bad MaxRM-RF predictions with mean of Ytr
            infty = ~np.isfinite(pred_maxrmrf)
            if np.any(infty):
                mean_ytr = float(np.mean(Ytr))
                pred_maxrmrf[infty] = mean_ytr

            infty = ~np.isfinite(pred_maxrmrf_revert)
            if np.any(infty):
                mean_ytr = float(np.mean(Ytr))
                pred_maxrmrf_revert[infty] = mean_ytr
            # ---------------------------------------------------------------

            # 8. Evaluate worst-case risk across target envs
            if risk_eval == "mse":
                max_risks[sim, 0] = max_mse(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf_revert, Ete)

            elif risk_eval == "reward":
                max_risks[sim, 0] = -min_reward(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf_revert, Ete)

            elif risk_eval == "regret":
                max_risks[sim, 0] = max_regret(
                    Yte, pred_maxrmrf, pred_erm, Ete
                )
                max_risks[sim, 1] = max_regret(
                    Yte, pred_maxrmrf_revert, pred_erm, Ete
                )

        # Aggregate replicate results into results dict
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label}), reverted"] = max_risks[
            :, 1
        ].tolist()

    np.save(
        os.path.join(
            OUT_DIR,
            f"opt{risk_label}_eval{risk_eval}_changeXdistr{str(change_X_distr)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.npy",
        ),
        results,
    )

    plot_maxrisk_vs_nenvs(
        results,
        change_X_distr,
        # ylim=(0.7, 1.35),
        risk_eval=risk_eval,
        risk_label=risk_label,
    )

    plot_indeterminate_prop_vs_nenvs(
        total_indeterminate_prop,
        Ls,
        change_X_distr,
        risk_eval=risk_eval,
        risk_label=risk_label,
    )
