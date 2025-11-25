# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP import DataContainer
from nldg.utils import max_mse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_rel

CHANGE_X_DISTR = True
OBS_PER_ENV = 2000
SIGMA = 0.25
MAX_NOISE_LEVEL = 5

N_SIM = 100
N_ESTIMATORS = 100
NUM_COVARIATES = 5
BETA_LOW = 0.5
BETA_HIGH = 2.5
N_JOBS = 5
L = 5
MIN_SAMPLES_LEAF = 30
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = os.path.join(OUT_DIR, "comparison_different_noise")
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # Columns:
    # 0: Oracle RF, 1: RF,
    # 2: Oracle MaxRM-RF(mse), 3: MaxRM-RF(mse),
    # 4: Oracle MaxRM-RF(reg), 5: MaxRM-RF(reg),
    # 6: Hack 1 (knowledge distillation),
    # 7: Hack 2(med), 8: Hack 2(min),
    # 9: Hack 2(sigma)
    max_risks = np.zeros((N_SIM, 10))

    for sim in tqdm(range(N_SIM), leave=False):
        data = DataContainer(
            n=OBS_PER_ENV,
            N=OBS_PER_ENV,
            L=L,
            d=NUM_COVARIATES,
            change_X_distr=CHANGE_X_DISTR,
            beta_low=BETA_LOW,
            beta_high=BETA_HIGH,
        )

        data.sigma_eps = SIGMA

        # resample functions for each simulation
        data.generate_dataset(seed=sim)

        stds = rng.uniform(0, MAX_NOISE_LEVEL, size=L)
        scales = np.repeat(stds, 2000)

        noise = rng.normal(size=2000 * L) * scales

        Xtr = np.vstack(data.X_sources_list)
        Ytr = np.concatenate(data.Y_sources_list)
        Ytr_noisy = Ytr + noise
        Etr = np.concatenate(data.E_sources_list)
        Xte = np.vstack(data.X_target_list)
        Yte = np.concatenate(data.Y_target_potential_list)
        Ete = np.concatenate(data.E_target_potential_list)

        # ERM solutions (noisy) for regret-based MaxRM-RF
        fitted_erm = np.zeros(len(Etr))
        fitted_erm_trees = np.zeros((N_ESTIMATORS, len(Etr)))

        # Oracle ERM solutions (clean labels) for oracle regret-based MaxRM-RF
        fitted_erm_oracle = np.zeros(len(Etr))
        fitted_erm_trees_oracle = np.zeros((N_ESTIMATORS, len(Etr)))

        for env in np.unique(Etr):
            mask = Etr == env
            Xtr_env = Xtr[mask]
            # Noisy ERM
            Ytr_env_noisy = Ytr_noisy[mask]
            rf_e = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf_e.fit(Xtr_env, Ytr_env_noisy)
            fitted_erm[mask] = rf_e.predict(Xtr_env)
            for i in range(N_ESTIMATORS):
                fitted_erm_trees[i, mask] = rf_e.trees[i].predict(Xtr_env)

            # Oracle ERM (clean labels)
            Ytr_env_clean = Ytr[mask]
            rf_e_or = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=SEED,
                n_jobs=N_JOBS,
            )
            rf_e_or.fit(Xtr_env, Ytr_env_clean)
            fitted_erm_oracle[mask] = rf_e_or.predict(Xtr_env)
            for i in range(N_ESTIMATORS):
                fitted_erm_trees_oracle[i, mask] = rf_e_or.trees[i].predict(
                    Xtr_env
                )

        # RF ------------------------------------------------------------
        rf_oracle = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf_oracle.fit(Xtr, Ytr)
        pred_rf_oracle = rf_oracle.predict(Xte)

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Ytr_noisy)
        pred_rf = rf.predict(Xte)
        # ---------------------------------------------------------------

        # MaxRM-RF(mse) -------------------------------------------------
        print("MaxRM-RF MSE")
        solvers = ["ECOS", "CLARABEL"]
        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_maxrmrf_mse = rf.predict(Xte)
        # ---------------------------------------------------------------

        # Oracle MaxRM-RF(mse) using clean labels -----------------------
        print("MaxRM-RF MSE (Oracle)")
        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf_oracle.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf_oracle.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_maxrmrf_mse_oracle = rf_oracle.predict(Xte)
        # ---------------------------------------------------------------

        # MaxRM-RF(reg) -------------------------------------------------
        print("MaxRM-RF Regret")
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Ytr_noisy)

        success = False
        kwargs = {
            "n_jobs": N_JOBS,
            "sols_erm": fitted_erm,
            "sols_erm_trees": fitted_erm_trees,
        }
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="regret",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="regret",
                **kwargs,
                opt_method="extragradient",
            )
        pred_maxrmrf_reg = rf.predict(Xte)
        # ----------------------------------------------------------------

        # Oracle MaxRM-RF(reg) using clean labels ------------------------
        print("MaxRM-RF Regret (Oracle)")
        rf_oracle = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf_oracle.fit(Xtr, Ytr)
        success = False
        kwargs = {
            "n_jobs": N_JOBS,
            "sols_erm": fitted_erm_oracle,
            "sols_erm_trees": fitted_erm_trees_oracle,
        }
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf_oracle.modify_predictions_trees(
                    Etr,
                    method="regret",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf_oracle.modify_predictions_trees(
                Etr,
                method="regret",
                **kwargs,
                opt_method="extragradient",
            )
        pred_maxrmrf_reg_oracle = rf_oracle.predict(Xte)
        # ---------------------------------------------------------------

        # hack 1: knowledge distillation --------------------------------
        print("Hack 1: knowledge distillation")
        Y_oob_preds = np.zeros(len(Etr))
        for env in np.unique(Etr):
            mask = Etr == env
            Xtr_env = Xtr[mask]
            Ytr_env_noisy = Ytr_noisy[mask]
            rf_e = RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                oob_score=True,
                bootstrap=True,
                n_jobs=N_JOBS,
                random_state=SEED,
            )
            rf_e.fit(Xtr_env, Ytr_env_noisy)
            Y_oob_preds[mask] = rf_e.oob_prediction_

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Y_oob_preds)

        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_hack1 = rf.predict(Xte)
        # ---------------------------------------------------------------

        # Hack 2(med) ---------------------------------------------------
        print("Hack 2(med)")
        v = np.zeros(len(np.unique(Etr)))

        for env in np.unique(Etr):
            mask = Etr == env
            v[env] = np.var(Ytr_noisy[mask] - Y_oob_preds[mask])

        a = np.sqrt(np.median(v)) / (np.sqrt(v) + 1e-8)

        Y_hack2_med = (1 - a[Etr]) * Y_oob_preds + a[Etr] * Ytr_noisy

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Y_hack2_med)

        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_hack2_med = rf.predict(Xte)
        # ---------------------------------------------------------------

        # Hack 2(min) ---------------------------------------------------
        print("Hack 2(min)")
        a = np.sqrt(np.min(v)) / (np.sqrt(v) + 1e-8)

        Y_hack2_min = (1 - a[Etr]) * Y_oob_preds + a[Etr] * Ytr_noisy

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Y_hack2_min)

        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_hack2_min = rf.predict(Xte)
        # ---------------------------------------------------------------

        # Hack 2(sigma) ---------------------------------------------------
        print("Hack 2(sigma)")
        a = data.sigma_eps / (np.sqrt(v) + 1e-8)

        Y_hack2_sigma = (1 - a[Etr]) * Y_oob_preds + a[Etr] * Ytr_noisy

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf.fit(Xtr, Y_hack2_sigma)

        success = False
        kwargs = {"n_jobs": N_JOBS}
        for solver in solvers:
            print("Trying solver:", solver)
            try:
                rf.modify_predictions_trees(
                    Etr,
                    method="mse",
                    **kwargs,
                    solver=solver,
                )
                success = True
                break
            except Exception:
                pass
        if not success:
            print("All solvers failed, switching to extragradient method.")
            rf.modify_predictions_trees(
                Etr,
                method="mse",
                **kwargs,
                opt_method="extragradient",
            )
        pred_hack2_sigma = rf.predict(Xte)
        # ---------------------------------------------------------------

        # Evaluate the maximum risk
        max_risks[sim, 0] = max_mse(Yte, pred_rf_oracle, Ete)
        max_risks[sim, 1] = max_mse(Yte, pred_rf, Ete)
        max_risks[sim, 2] = max_mse(Yte, pred_maxrmrf_mse_oracle, Ete)
        max_risks[sim, 3] = max_mse(Yte, pred_maxrmrf_mse, Ete)
        max_risks[sim, 4] = max_mse(Yte, pred_maxrmrf_reg_oracle, Ete)
        max_risks[sim, 5] = max_mse(Yte, pred_maxrmrf_reg, Ete)
        max_risks[sim, 6] = max_mse(Yte, pred_hack1, Ete)
        max_risks[sim, 7] = max_mse(Yte, pred_hack2_med, Ete)
        max_risks[sim, 8] = max_mse(Yte, pred_hack2_min, Ete)
        max_risks[sim, 9] = max_mse(Yte, pred_hack2_sigma, Ete)

    t_stat, p_val = ttest_rel(max_risks[:, 5], max_risks[:, 6])
    print(f"Paired t-test MaxRM-RF(reg) vs Hack 1: p-value = {p_val:.4e}")

    t_stat, p_val = ttest_rel(max_risks[:, 5], max_risks[:, 7])
    print(f"Paired t-test MaxRM-RF(reg) vs Hack 2(med): p-value = {p_val:.4e}")

    t_stat, p_val = ttest_rel(max_risks[:, 5], max_risks[:, 8])
    print(f"Paired t-test MaxRM-RF(reg) vs Hack 2(min): p-value = {p_val:.4e}")

    t_stat, p_val = ttest_rel(max_risks[:, 5], max_risks[:, 9])
    print(
        f"Paired t-test MaxRM-RF(reg) vs Hack 2(sigma): p-value = {p_val:.4e}"
    )

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        [
            max_risks[:, 0],  # Oracle RF
            max_risks[:, 1],  # RF
            max_risks[:, 2],  # Oracle MaxRM-RF(mse)
            max_risks[:, 3],  # MaxRM-RF(mse)
            max_risks[:, 4],  # Oracle MaxRM-RF(reg)
            max_risks[:, 5],  # MaxRM-RF(reg)
            max_risks[:, 6],  # Hack 1 (knowledge distillation)
            max_risks[:, 7],  # Hack 2(med)
            max_risks[:, 8],  # Hack 2(min)
            max_risks[:, 9],  # Hack 2(sigma)
        ],
        tick_labels=[
            "Oracle\nRF",
            "RF",
            "Oracle\nMaxRM-RF(mse)",
            "MaxRM-RF(mse)",
            "Oracle\nMaxRM-RF(reg)",
            "MaxRM-RF(reg)",
            "Hack 1",
            "Hack 2(med)",
            "Hack 2(min)",
            "Hack 2(sigma)",
        ],
        patch_artist=True,
    )
    ax.tick_params(axis="x", labelsize=5)
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def lighten(color_hex, amount):
        from matplotlib.colors import to_rgb

        c = np.array(to_rgb(color_hex))
        return tuple(c + (1 - c) * amount)

    colors = [
        lighten(base_colors[0], 0.6),  # Oracle RF (lighter blue)
        lighten(base_colors[0], 0.3),  # RF (darker blue)
        lighten(base_colors[1], 0.6),  # Oracle MSE (lighter orange)
        lighten(base_colors[1], 0.3),  # MSE (darker orange)
        lighten(base_colors[2], 0.6),  # Oracle Regret (lighter green)
        lighten(base_colors[2], 0.3),  # Regret (darker green)
        lighten(base_colors[3], 0.5),  # Hack 1 (medium red)
        lighten(base_colors[4], 0.5),  # Hack 2(med) (medium purple)
        lighten(base_colors[4], 0.5),  # Hack 2(min) (medium purple)
        lighten(base_colors[4], 0.5),  # Hack 2(sigma) (medium purple)
    ]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
    ax.set_ylabel("Maximum MSE across environments")
    ax.set_title("Comparison of methods under different noise levels")
    ax.yaxis.grid(True, which="major", linestyle="--")
    plt.tight_layout()

    base_fname = f"boxplot_changeXdistr{str(CHANGE_X_DISTR)}_reps{N_SIM}_obsPerEnv{OBS_PER_ENV}_sigma{SIGMA}"
    plt.savefig(os.path.join(OUT_DIR, f"{base_fname}.png"), dpi=300)
    np.save(os.path.join(OUT_DIR, f"{base_fname}.npy"), max_risks)
    plt.close(fig)
