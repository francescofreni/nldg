"""This script compares standard Random Forests (RF) with weighted Random Forests (W-RF)
using weights fitted via empirical risk minimization (ERM). The comparison is done
in terms of out-of-sample mean squared error (MSE) on regression tasks with varying
training sample sizes. In this experiments, there are no distributional shifts."""

import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data import DataContainer
from tqdm import tqdm
import cvxpy as cp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def fit_weights(pred, y):
    """Fit weights w to minimize ||y - preds @ w||^2
    under the constraints sum(w) = 1 and w >= 0.

    Args:
        pred: matrix of per-tree predictions, shape (n_samples, n_trees)
        y: response vector, shape (n_samples,)

    Returns:
        w: fitted weights, shape (n_trees,)
    """
    d = pred.shape[1]
    w = cp.Variable(d)

    objective = cp.Minimize(cp.sum_squares(pred @ w - y))
    cons = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.ECOS)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"ECOS status = {prob.status}, retrying with SCS.")
            prob.solve(solver=cp.SCS)
    except cp.SolverError:
        print("ECOS failed — retrying with SCS.")
        prob.solve(solver=cp.SCS)
    return w.value


N_SIM = 25
N_VALUES = [1000, 2500, 5000, 10000, 15000, 20000]
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
SEED = 42
N_JOBS = 12
NUM_COVARIATES = 10


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_weights_ERM")
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":
    results_RF = pd.DataFrame(
        0, index=range(N_SIM), columns=N_VALUES, dtype=float
    )
    results_WRF1 = pd.DataFrame(
        0, index=range(N_SIM), columns=N_VALUES, dtype=float
    )
    results_WRF2 = pd.DataFrame(
        0, index=range(N_SIM), columns=N_VALUES, dtype=float
    )

    results_RF_insample = pd.DataFrame(
        0, index=range(N_SIM), columns=N_VALUES, dtype=float
    )
    results_WRF1_insample = pd.DataFrame(
        0, index=range(N_SIM), columns=N_VALUES, dtype=float
    )

    for n in tqdm(N_VALUES, leave=False):
        print(f"Running experiments for n={n}...")

        # generate 5*n iid samples, use n for training and rest for testing
        data = DataContainer(
            n=5 * n,
            N=10000,
            d=NUM_COVARIATES,
            change_X_distr=False,
            risk="mse",
        )

        data.generate_funcs_list(L=1, seed=SEED)

        for sim in range(N_SIM):
            print(f" Simulation {sim + 1}/{N_SIM}")
            data.generate_data(seed=sim)

            Xtr = np.vstack(data.X_sources_list)
            Ytr = np.concatenate(data.Y_sources_list)

            # use last 4*n samples as test set
            Xte = Xtr[n:, :].copy()
            Yte = Ytr[n:].copy()

            # use first n samples as training set
            Xtr = Xtr[:n, :].copy()
            Ytr = Ytr[:n].copy()

            # for weighted random forests with out-of-sample predictions
            Xtr_t, Xtr_w, Ytr_t, Ytr_w = train_test_split(
                Xtr, Ytr, test_size=0.3, random_state=SEED, shuffle=True
            )

            # fit random forests ------------------------------------------
            rf_1 = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf_1.fit(Xtr, Ytr)
            pred_rf_1 = rf_1.predict(Xte)
            pred_rf_1_insample = rf_1.predict(Xtr)

            # per-tree predictions
            pred_trees_rf_1 = np.column_stack(
                [tree.predict(Xtr) for tree in rf_1.trees]
            )

            rf_2 = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf_2.fit(Xtr_t, Ytr_t)

            # per-tree predictions
            pred_trees_rf_2 = np.column_stack(
                [tree.predict(Xtr_w) for tree in rf_2.trees]
            )
            # ---------------------------------------------------------------

            # weighted versions ---------------------------------------------

            weights_1 = fit_weights(pred_trees_rf_1, Ytr)
            # out-of-sample predictions
            pred_wrf_1 = (
                np.column_stack([tree.predict(Xte) for tree in rf_1.trees])
                @ weights_1
            )
            # in-sample predictions
            pred_wrf_1_insample = pred_trees_rf_1 @ weights_1

            weights_2 = fit_weights(pred_trees_rf_2, Ytr_w)
            # out-of-sample predictions
            pred_wrf_2 = (
                np.column_stack([tree.predict(Xte) for tree in rf_2.trees])
                @ weights_2
            )

            # ---------------------------------------------------------------

            # compute out-of-sample MSEs
            results_RF.loc[sim, n] = np.mean((Yte - pred_rf_1) ** 2)
            results_WRF1.loc[sim, n] = np.mean((Yte - pred_wrf_1) ** 2)
            results_WRF2.loc[sim, n] = np.mean((Yte - pred_wrf_2) ** 2)

            # compute in-sample MSEs
            results_RF_insample.loc[sim, n] = np.mean(
                (Ytr - pred_rf_1_insample) ** 2
            )
            results_WRF1_insample.loc[sim, n] = np.mean(
                (Ytr - pred_wrf_1_insample) ** 2
            )

    # -------------------------------------------------------------------------
    # Plot: MSE vs number of training samples with 95% t-intervals across sims

    def _summarize_with_t_ci(df: pd.DataFrame):
        """Return x (n values), mean MSE, and 95% t CI half-width per n."""
        # Ensure x is sorted by n
        x = np.array(sorted(df.columns))
        arr = df[x].to_numpy()
        means = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)
        n_sims = arr.shape[0]
        tcrit = stats.t.ppf(0.975, df=n_sims - 1)
        half = tcrit * (std / np.sqrt(n_sims))
        return x, means, half

    fig, ax = plt.subplots(figsize=(7, 5))

    series = [
        ("RF", results_RF, "#5790FC"),
        ("W-RF-1", results_WRF1, "#964A8B"),
        ("W-RF-2", results_WRF2, "#FF61A6"),
    ]

    for label, df, color in series:
        x, mean_mse, ci = _summarize_with_t_ci(df)
        ax.plot(x, mean_mse, label=label, color=color, marker="o")
        ax.fill_between(
            x, mean_mse - ci, mean_mse + ci, color=color, alpha=0.2
        )

    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("MSE")
    ax.set_title(
        f"Out-of-sample MSE vs training size (95% t-test CI over {N_SIM} reps)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"comparison_weights_ERM.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
