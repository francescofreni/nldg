# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP import DataContainer
from nldg.utils import min_reward, max_mse, max_regret
from nldg.additional.gdro import GroupDRO
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


N_SIM = 25
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
SEED = 42
COLORS = {
    "RF": "#5790FC",
    "MaxRM-RF": "#F89C20",
    "Weighted RF": "#964A8B",
    "Weighted MaxRM-RF": "#28A745",
    "GroupDRO-NN": "#D62728",
}

NUM_COVARIATES = 10
CHANGE_X_DISTR = False
risk = "regret"  # "mse", "reward", "regret"
risk_label = "reg"  # "mse", "nrw", "reg"
N_JOBS = 20
COMMON_CORE_FUNC = False

# number of environments
Ls = [3, 4, 5, 6, 7, 8, 9, 10]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_weights_MaxRM")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_maxrisk_vs_nenvs(
    results: dict[int, dict[str, float]],
    risk_label: str = "mse",
    change_X_distr: bool = False,
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
            f"{risk_label}_comparison_weights_MaxRM_changeXdistr{str(change_X_distr)}_commonCoreFunc{str(COMMON_CORE_FUNC)}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    results = {L: {} for L in Ls}

    for L in tqdm(Ls):
        data = DataContainer(
            n=2000,
            N=2000,
            d=NUM_COVARIATES,
            change_X_distr=CHANGE_X_DISTR,
            risk=risk,
            target_mode="convex_mixture_P",
            common_core_func=COMMON_CORE_FUNC,
        )

        data.generate_funcs_list(L=L, seed=SEED)
        max_risks = np.zeros((N_SIM, 3))

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
                        n_jobs=N_JOBS,
                    )
                    rf_e.fit(Xtr_env, Ytr_env)
                    fitted_erm[mask] = rf_e.predict(Xtr_env)
                    for i in range(N_ESTIMATORS):
                        fitted_erm_trees[i, mask] = rf_e.trees[i].predict(
                            Xtr_env
                        )
                    mask_te = Ete == env
                    pred_erm[mask_te] = rf_e.predict(Xte[mask_te])

            # RF ------------------------------------------------------------
            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf.fit(Xtr, Ytr)
            pred_rf = rf.predict(Xte)

            # ---------------------------------------------------------------

            # GroupDRO-NN ---------------------------------------------------
            gdro = GroupDRO(
                data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk=risk
            )
            gdro.fit(epochs=500)
            pred_gdro = gdro.predict(Xte)

            if risk == "regret":
                pred_erm_gdro = gdro.predict_per_group(Xte, Ete)

            # ---------------------------------------------------------------

            # MaxRM-RF ------------------------------------------------------
            solvers = ["ECOS", "SCS"]
            success = False
            kwargs = {"n_jobs": N_JOBS}
            if risk == "regret":
                kwargs["sols_erm"] = fitted_erm
                kwargs["sols_erm_trees"] = fitted_erm_trees
            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr,
                        method="reward",
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
                    method="reward",
                    **kwargs,
                    opt_method="extragradient",
                )
            pred_maxrmrf = rf.predict(Xte)

            # ---------------------------------------------------------------

            # Evaluate the maximum risk

            if risk == "mse":
                max_risks[sim, 0] = max_mse(Yte, pred_rf, Ete)
                max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = max_mse(Yte, pred_gdro, Ete)
            elif risk == "reward":
                max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
                max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = -min_reward(Yte, pred_gdro, Ete)
            else:
                max_risks[sim, 0] = max_regret(Yte, pred_rf, pred_erm, Ete)
                max_risks[sim, 1] = max_regret(
                    Yte, pred_maxrmrf, pred_erm, Ete
                )
                max_risks[sim, 2] = max_regret(
                    Yte, pred_gdro, pred_erm_gdro, Ete
                )

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"GroupDRO-NN({risk_label})"] = max_risks[:, 2].tolist()

    np.save(
        f"({risk_label})_results_changeXdistr_{CHANGE_X_DISTR}_commonCoreFunc_{COMMON_CORE_FUNC}.npy",
        results,
    )

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
        change_X_distr=CHANGE_X_DISTR,
    )
