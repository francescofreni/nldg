# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP_proper import DataContainer
from nldg.additional.gdro import GroupDRO
from nldg.rf import MaggingRF
from nldg.utils import max_mse
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


N_SIM = 100
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 15
SEED = 42
COLORS = {
    "RF": "#5790FC",
    "MaxRM-RF": "#F89C20",
    "GroupDRO-NN": "#86C8DD",
    "Magging-RF": "#964A8B",
}

NUM_COVARIATES = 5
CHANGE_X_DISTR = False
BETA_LOW = 0.5
BETA_HIGH = 2.5

risk_label = "mse"  # "mse", "nrw", "reg"

N_JOBS = 5

# number of observations per environment
N_vec = [50, 100, 200, 300, 400, 500]

# number of environments (that are equal)
L = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_equal_envs")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_maxrisk_vs_nenvs(
    results: dict[int, dict[str, float]],
    risk_label: str = "mse",
) -> None:
    fontsize = 15

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

    ax.set_xlabel("Number of observations per environment", fontsize=fontsize)
    ax.set_ylabel("Maximum MSE across environments", fontsize=fontsize)
    ax.set_xticks(L_vals)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=True, fontsize=fontsize)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{risk_label}_equalenv_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    results = {L: {} for L in N_vec}

    for n in tqdm(N_vec):
        max_risks = np.zeros((N_SIM, 4))

        for sim in tqdm(range(N_SIM), leave=False):
            data = DataContainer(
                n=L * n,
                N=L * n,
                L=1,
                d=NUM_COVARIATES,
                change_X_distr=CHANGE_X_DISTR,
                risk="mse",
                beta_low=BETA_LOW,
                beta_high=BETA_HIGH,
            )

            # resample functions for each simulation
            data.generate_dataset(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)
            Xte = np.vstack(data.X_target_list)
            Yte = np.concatenate(data.Y_target_potential_list)

            # split the iid data into L "environments"
            Etr = np.repeat(np.arange(5), n)
            Ete = np.repeat(np.arange(5), n)

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
                min_samples_leaf=MIN_SAMPLES_LEAF // L,
                random_state=SEED,
                backend="adaXT",
                risk="mse",
            )
            rf_magging.fit(Xtr, Ytr, Etr)
            pred_magging = rf_magging.predict(Xte)
            # ---------------------------------------------------------------

            # GroupDRO-NN ---------------------------------------------------
            gdro = GroupDRO(
                data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk="mse"
            )
            gdro.fit(epochs=500)
            pred_gdro = gdro.predict(Xte)
            # ---------------------------------------------------------------

            # MaxRM-RF ------------------------------------------------------
            solvers = ["ECOS", "SCS", "CLARABEL"]
            success = False
            kwargs = {"n_jobs": N_JOBS}
            for solver in solvers:
                try:
                    rf.modify_predictions_trees(
                        Etr,
                        method="mse",
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
                    method="mse",
                    **kwargs,
                    opt_method="extragradient",
                )
            pred_maxrmrf = rf.predict(Xte)
            # ---------------------------------------------------------------

            # Evaluate the risk
            max_risks[sim, 0] = max_mse(Yte, pred_rf, Ete)
            max_risks[sim, 1] = max_mse(Yte, pred_maxrmrf, Ete)
            max_risks[sim, 2] = max_mse(Yte, pred_gdro, Ete)
            max_risks[sim, 3] = max_mse(Yte, pred_magging, Ete)

        results[n]["RF"] = max_risks[:, 0].tolist()
        results[n][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[n][f"GroupDRO-NN({risk_label})"] = max_risks[:, 2].tolist()
        results[n][f"Magging-RF({risk_label})"] = max_risks[:, 3].tolist()

    np.save(
        os.path.join(
            OUT_DIR,
            f"{risk_label}_equalenv_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.npy",
        ),
        results,
    )

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
    )
