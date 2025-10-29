# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP import DataContainer
from nldg.utils import min_reward
from tqdm import tqdm
from experiments.additional.comparison_gdro import plot_maxrisk_vs_nenvs
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


N_SIM = 10
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
SEED = 42
COLORS = {
    "RF": "#5790FC",
    "MaxRM-RF": "#F89C20",
    "W-RF-1": "#FF61A6",
    "WRF": "#964A8B",
    "W-MaxRM-RF-1": "#F00000",
    "MaxRM-WRF": "#28A745",
}

NUM_COVARIATES = 10
CHANGE_X_DISTR = False
UNBALANCED_ENVS = False
risk_label = "nrw"
N_JOBS = 12

# number of environments
Ls = [3, 5, 7, 9]

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
            f"{loss_type}_comparison_weights_MaxRM_changeXdistr{str(change_X_distr)}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )


def rank_of_best_tree(randomforest, Xte, Yte, Ete, weights):
    """Compute the rank of the weight of the best out-of-sample tree."""
    pred_rf_trees = [tree.predict(Xte) for tree in randomforest.trees]
    rewards = np.array([-min_reward(Yte, pred, Ete) for pred in pred_rf_trees])

    # find rank of the weight of best out-of-sample tree
    idx = np.argmin(rewards)
    ranks = weights.value.argsort()[::-1].argsort() + 1
    return ranks[idx]


if __name__ == "__main__":
    num_of_zero_weights_wrf1 = []
    num_of_zero_weights_wrf2 = []
    num_of_zero_weights_maxrmf1 = []
    num_of_zero_weights_maxrmf2 = []

    rank_of_best_trees_wrf1 = []
    rank_of_best_trees_wrf2 = []
    rank_of_best_trees_maxrmf1 = []
    rank_of_best_trees_maxrmf2 = []

    results = {L: {} for L in Ls}
    results_insample = {L: {} for L in Ls}

    for L in tqdm(Ls):
        if not UNBALANCED_ENVS:
            data = DataContainer(
                n=2000,
                N=2000,
                d=NUM_COVARIATES,
                change_X_distr=CHANGE_X_DISTR,
                risk="reward",
            )
        else:
            data = DataContainer(
                n=1000,
                N=1000,
                d=NUM_COVARIATES,
                change_X_distr=CHANGE_X_DISTR,
                risk="reward",
                unbalanced_envs=UNBALANCED_ENVS,
            )

        data.generate_funcs_list(n_E=L, seed=SEED)
        max_risks = np.zeros((N_SIM, 6))
        max_risks_insample = np.zeros((N_SIM, 6))

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

            # in-sample predictions
            pred_trees_rf_insample = np.column_stack(
                [tree.predict(Xtr) for tree in rf.trees]
            )
            pred_rf_insample = pred_trees_rf_insample.mean(axis=1)
            # ---------------------------------------------------------------

            # W-RF-1 --------------------------------------------------------
            pred_wrf1, weights_wrf1 = rf.refine_weights(
                X_val=Xtr,
                Y_val=Ytr,
                E_val=Etr,
                X=Xte,
                risk=risk_label,
            )

            # in-sample predictions
            pred_wrf1_insample = pred_trees_rf_insample @ weights_wrf1.value

            # number of zero weights
            num_of_zero_weights_wrf1.append(np.sum(weights_wrf1.value == 0.0))

            # rank of best tree
            rank_of_best_trees_wrf1.append(
                rank_of_best_tree(rf, Xte, Yte, Ete, weights_wrf1)
            )
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

            # in-sample predictions
            pred_maxrmrf_insample = rf.predict(Xtr)
            # ---------------------------------------------------------------

            # W-MaxRM-RF-1 --------------------------------------------------
            pred_w_maxrmf_1, weights_w_maxrmf_1 = rf.refine_weights(
                X_val=Xtr,
                Y_val=Ytr,
                E_val=Etr,
                X=Xte,
                risk=risk_label,
            )

            # in-sample predictions
            pred_w_maxrmf_1_insample = (
                np.column_stack([tree.predict(Xtr) for tree in rf.trees])
                @ weights_w_maxrmf_1.value
            )

            # number of zero weights
            num_of_zero_weights_maxrmf1.append(
                np.sum(weights_w_maxrmf_1.value == 0.0)
            )

            # rank of best tree
            rank_of_best_trees_maxrmf1.append(
                rank_of_best_tree(rf, Xte, Yte, Ete, weights_w_maxrmf_1)
            )
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

            # in-sample predictions
            pred_wrf2_insample = (
                np.column_stack([tree.predict(Xtr) for tree in rf_t.trees])
                @ weights_wrf2.value
            )

            # number of zero weights
            num_of_zero_weights_wrf2.append(np.sum(weights_wrf2.value == 0.0))

            # rank of best tree
            rank_of_best_trees_wrf2.append(
                rank_of_best_tree(rf_t, Xte, Yte, Ete, weights_wrf2)
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

            # in-sample predictions
            pred_w_maxrmf_2_insample = (
                np.column_stack([tree.predict(Xtr) for tree in rf_t.trees])
                @ weights_w_maxrmf_2.value
            )

            # number of zero weights
            num_of_zero_weights_maxrmf2.append(
                np.sum(weights_w_maxrmf_2.value == 0.0)
            )

            # rank of best tree
            rank_of_best_trees_maxrmf2.append(
                rank_of_best_tree(rf_t, Xte, Yte, Ete, weights_w_maxrmf_2)
            )
            # ---------------------------------------------------------------

            # Evaluate the maximum risk
            max_risks[sim, 0] = -min_reward(Yte, pred_rf, Ete)
            max_risks[sim, 1] = -min_reward(Yte, pred_maxrmrf, Ete)
            max_risks[sim, 2] = -min_reward(Yte, pred_wrf1, Ete)
            max_risks[sim, 3] = -min_reward(Yte, pred_wrf2, Ete)
            max_risks[sim, 4] = -min_reward(Yte, pred_w_maxrmf_1, Ete)
            max_risks[sim, 5] = -min_reward(Yte, pred_w_maxrmf_2, Ete)

            # in-sample risks
            max_risks_insample[sim, 0] = -min_reward(
                Ytr, pred_rf_insample, Etr
            )
            max_risks_insample[sim, 1] = -min_reward(
                Ytr, pred_maxrmrf_insample, Etr
            )
            max_risks_insample[sim, 2] = -min_reward(
                Ytr, pred_wrf1_insample, Etr
            )
            max_risks_insample[sim, 3] = -min_reward(
                Ytr, pred_wrf2_insample, Etr
            )
            max_risks_insample[sim, 4] = -min_reward(
                Ytr, pred_w_maxrmf_1_insample, Etr
            )
            max_risks_insample[sim, 5] = -min_reward(
                Ytr, pred_w_maxrmf_2_insample, Etr
            )

        results[L]["RF"] = max_risks[:, 0].tolist()
        results[L][f"MaxRM-RF({risk_label})"] = max_risks[:, 1].tolist()
        results[L][f"W-RF-1({risk_label})"] = max_risks[:, 2].tolist()
        results[L][f"Weighted RF({risk_label})"] = max_risks[:, 3].tolist()
        results[L][f"W-MaxRM-RF-1({risk_label})"] = max_risks[:, 4].tolist()
        results[L][f"Weighted MaxRM-RF({risk_label})"] = max_risks[
            :, 5
        ].tolist()

        results_insample[L]["RF"] = max_risks_insample[:, 0].tolist()
        results_insample[L][f"MaxRM-RF({risk_label})"] = max_risks_insample[
            :, 1
        ].tolist()
        results_insample[L][f"W-RF-1({risk_label})"] = max_risks_insample[
            :, 2
        ].tolist()
        results_insample[L][f"Weighted RF({risk_label})"] = max_risks_insample[
            :, 3
        ].tolist()
        results_insample[L][f"W-MaxRM-RF-1({risk_label})"] = (
            max_risks_insample[:, 4].tolist()
        )
        results_insample[L][f"Weighted MaxRM-RF({risk_label})"] = (
            max_risks_insample[:, 5].tolist()
        )

    np.save(f"results_changeXdistr_{CHANGE_X_DISTR}.npy", results)
    np.save(
        f"results_insample_changeXdistr_{CHANGE_X_DISTR}.npy",
        results_insample,
    )

    print(np.mean(rank_of_best_trees_wrf1))
    print(np.mean(rank_of_best_trees_wrf2))
    print(np.mean(rank_of_best_trees_maxrmf1))
    print(np.mean(rank_of_best_trees_maxrmf2))

    print(np.mean(num_of_zero_weights_wrf1))
    print(np.mean(num_of_zero_weights_wrf2))
    print(np.mean(num_of_zero_weights_maxrmf1))
    print(np.mean(num_of_zero_weights_maxrmf2))

    plot_maxrisk_vs_nenvs(
        results,
        risk_label=risk_label,
        loss_type="out-of-sample",
        change_X_distr=CHANGE_X_DISTR,
        unbalanced_envs=UNBALANCED_ENVS,
    )

    plot_maxrisk_vs_nenvs(
        results_insample,
        risk_label=risk_label,
        loss_type="in-sample",
        change_X_distr=CHANGE_X_DISTR,
        unbalanced_envs=UNBALANCED_ENVS,
    )
