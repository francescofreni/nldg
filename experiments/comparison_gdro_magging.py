# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP_proper import DataContainer
from nldg.utils import min_reward, max_mse, max_regret
from nldg.additional.gdro import GroupDRO
from nldg.rf import MaggingRF
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


N_SIM = 100
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 30
SEED = 42
COLORS = {
    "RF": "#5790FC",
    "MaxRM-RF": "#F89C20",
    "Group DRO": "#86C8DD",
    "Magging": "#964A8B",
}

NUM_COVARIATES = 5
CHANGE_X_DISTR = True
BETA_LOW = 0.5
BETA_HIGH = 2.5

risk = "mse"  # "mse", "reward", "regret"
risk_label = "mse"  # "mse", "nrw", "reg"

risk_eval = "mse"  # "mse", "reward", "regret"
N_JOBS = 5

# number of environments
Ls = [2, 3, 4, 5, 6, 7, 8]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_gdro_magging")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_maxrisk_vs_nenvs(
    results: dict[int, dict[str, float]],
    risk_eval: str = "mse",
    risk_label: str = "mse",
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

    ax.set_xlabel("Number of environments $K$")
    if risk_eval == "mse":
        ax.set_ylabel("Maximum MSE across environments")
    elif risk_eval == "nrw":
        ax.set_ylabel("Maximum Negative Reward across environments")
    else:
        ax.set_ylabel("Maximum Regret across environments")

    ax.set_xticks(L_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"opt{risk_label}_eval{risk_eval}_changeXdistr{str(CHANGE_X_DISTR)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    results = {L: {} for L in Ls}

    for L in tqdm(Ls):
        max_risks = np.zeros((N_SIM, 4))

        for sim in tqdm(range(N_SIM), leave=False):
            data = DataContainer(
                n=2000,
                N=2000,
                L=L,
                d=NUM_COVARIATES,
                change_X_distr=CHANGE_X_DISTR,
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

            # Magging -------------------------------------------------------
            rf_magging = MaggingRF(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=SEED,
                backend="adaXT",
                risk=risk_label,
                sols_erm=fitted_erm,
            )
            fitted_magging = rf_magging.fit(Xtr, Ytr, Etr)
            pred_magging = rf_magging.predict(Xte)
            # ---------------------------------------------------------------

            # GroupDRO-NN ---------------------------------------------------
            gdro = GroupDRO(
                data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk=risk
            )
            gdro.fit(epochs=500)
            pred_gdro = gdro.predict(Xte)
            # ---------------------------------------------------------------

            # MaxRM-RF ------------------------------------------------------
            solvers = ["ECOS", "SCS", "CLARABEL"]
            success = False
            kwargs = {"n_jobs": N_JOBS}
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
            # ---------------------------------------------------------------

            # Evaluate the maximum risk

            if risk_eval == "mse":
                max_risks[sim, 0] = max_mse(Yte, pred_rf, Ete)
                max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = max_mse(Yte, pred_gdro, Ete)
                max_risks[sim, 3] = max_mse(Yte, pred_magging, Ete)

            elif risk_eval == "reward":
                max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
                max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
                max_risks[sim, 2] = -min_reward(Yte, pred_gdro, Ete)
                max_risks[sim, 3] = -min_reward(Yte, pred_magging, Ete)

            else:
                max_risks[sim, 0] = max_regret(Yte, pred_rf, pred_erm, Ete)
                max_risks[sim, 1] = max_regret(
                    Yte, pred_maxrmrf, pred_erm, Ete
                )
                max_risks[sim, 2] = max_regret(Yte, pred_gdro, pred_erm, Ete)
                max_risks[sim, 3] = max_regret(
                    Yte, pred_magging, pred_erm, Ete
                )

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"Group DRO({risk_label})"] = max_risks[:, 2].tolist()
        results[L][f"Magging({risk_label})"] = max_risks[:, 3].tolist()

    np.save(
        os.path.join(
            OUT_DIR,
            f"opt{risk_label}_eval{risk_eval}_changeXdistr{str(CHANGE_X_DISTR)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.npy",
        ),
        results,
    )

    plot_maxrisk_vs_nenvs(
        results,
        risk_eval=risk_eval,
        risk_label=risk_label,
    )
