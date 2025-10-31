# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP import DataContainer
from nldg.utils import min_reward
from nldg.additional.gdro import GroupDRO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


N_SIM = 15
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
CHANGE_X_DISTR = True
risk_label = "nrw"
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
    loss_type: str = "out-of-sample",
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
            f"{loss_type}_comparison_weights_MaxRM_changeXdistr{str(change_X_distr)}_commonCoreFunc{str(COMMON_CORE_FUNC)}.pdf",
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
            risk="reward",
            target_mode="convex_mixture_P",
        )

        data.generate_funcs_list(L=L, seed=SEED)
        max_risks = np.zeros((N_SIM, 5))

        for sim in tqdm(range(N_SIM), leave=False):
            data.generate_data(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)
            Etr = np.concatenate(data.E_sources_list)
            Xte = np.vstack(data.X_target_list)
            Yte = np.concatenate(data.Y_target_potential_list)
            Ete = np.concatenate(data.E_target_potential_list)

            # for weighted random forests with out-of-sample predictions
            Xtr_t, Xtr_w, Ytr_t, Ytr_w, Etr_t, Etr_w = train_test_split(
                Xtr, Ytr, Etr, test_size=0.3, random_state=SEED, shuffle=True
            )

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
                data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk="reward"
            )
            gdro.fit(epochs=500)
            pred_gdro = gdro.predict(Xte)

            # ---------------------------------------------------------------

            # MaxRM-RF ------------------------------------------------------
            solvers = ["ECOS", "SCS"]
            success = False
            kwargs = {"n_jobs": N_JOBS}
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

            # W-RF-2 --------------------------------------------------------

            # fit trees on a part of the training data
            rf_t = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf_t.fit(Xtr_t, Ytr_t)

            pred_wrf2, weights_wrf2 = rf_t.refine_weights(
                X_val=Xtr_w,
                Y_val=Ytr_w,
                E_val=Etr_w,
                X=Xte,
                risk=risk_label,
            )

            # ---------------------------------------------------------------

            # W-MaxRM-RF-2 --------------------------------------------------
            solvers = ["ECOS", "SCS"]
            success = False
            kwargs = {"n_jobs": N_JOBS}
            for solver in solvers:
                try:
                    rf_t.modify_predictions_trees(
                        Etr_t,
                        method="reward",
                        **kwargs,
                        solver=solver,
                    )
                    success = True
                    break
                except Exception as e_try:
                    pass
            if not success:
                rf_t.modify_predictions_trees(
                    Etr_t,
                    method="reward",
                    **kwargs,
                    opt_method="extragradient",
                )

            pred_w_maxrmf_2, weights_w_maxrmf_2 = rf_t.refine_weights(
                X_val=Xtr_w,
                Y_val=Ytr_w,
                E_val=Etr_w,
                X=Xte,
                risk=risk_label,
            )

            # ---------------------------------------------------------------

            # Evaluate the maximum risk
            max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
            max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
            max_risks[sim, 2] = -min_reward(Yte, pred_wrf2, Ete)
            max_risks[sim, 3] = -min_reward(Yte, pred_w_maxrmf_2, Ete)
            max_risks[sim, 4] = -min_reward(Yte, pred_gdro, Ete)

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"Weighted RF({risk_label})"] = max_risks[:, 2].tolist()
        results[L][f"Weighted MaxRM-RF({risk_label})"] = max_risks[
            :, 3
        ].tolist()
        results[L][f"GroupDRO-NN({risk_label})"] = max_risks[:, 4].tolist()

    np.save(
        f"results_changeXdistr_{CHANGE_X_DISTR}_commonCoreFunc_{COMMON_CORE_FUNC}.npy",
        results,
    )

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
        loss_type="out-of-sample",
        change_X_distr=CHANGE_X_DISTR,
    )
