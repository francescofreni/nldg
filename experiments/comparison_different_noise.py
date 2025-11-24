# code modified from https://github.com/zywang0701/DRoL/blob/main/simu1.py
import os
import numpy as np
from adaXT.random_forest import RandomForest
from nldg.additional.data_GP_proper import DataContainer
from nldg.utils import max_mse
from tqdm import tqdm
import matplotlib.pyplot as plt


N_SIM = 20
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 30
SEED = 42

NUM_COVARIATES = 5
CHANGE_X_DISTR = True
BETA_LOW = 0.5
BETA_HIGH = 2.5

N_JOBS = 10

# number of environments
L = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
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
    # 4: Oracle MaxRM-RF(reg), 5: MaxRM-RF(reg)
    max_risks = np.zeros((N_SIM, 6))

    for sim in tqdm(range(N_SIM), leave=False):
        data = DataContainer(
            n=2000,
            N=2000,
            L=L,
            d=NUM_COVARIATES,
            change_X_distr=CHANGE_X_DISTR,
            beta_low=BETA_LOW,
            beta_high=BETA_HIGH,
        )

        # resample functions for each simulation
        data.generate_dataset(seed=sim)

        stds = rng.uniform(0, 5, size=L)
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

        # MaxRM-RF ------------------------------------------------------
        print("MaxRM-RF MSE")
        solvers = ["ECOS", "CLARABEL", "SCS"]
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
        # Oracle MaxRM-RF (mse) using clean labels ----------------------
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

        # MaxRM-RF ------------------------------------------------------
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
        # Oracle MaxRM-RF (regret) using clean labels -------------------
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

        # Evaluate the maximum risk
        max_risks[sim, 0] = max_mse(Yte, pred_rf_oracle, Ete)
        max_risks[sim, 1] = max_mse(Yte, pred_rf, Ete)
        max_risks[sim, 2] = max_mse(Yte, pred_maxrmrf_mse_oracle, Ete)
        max_risks[sim, 3] = max_mse(Yte, pred_maxrmrf_mse, Ete)
        max_risks[sim, 4] = max_mse(Yte, pred_maxrmrf_reg_oracle, Ete)
        max_risks[sim, 5] = max_mse(Yte, pred_maxrmrf_reg, Ete)

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        [
            max_risks[:, 0],  # Oracle RF
            max_risks[:, 1],  # RF
            max_risks[:, 2],  # Oracle MaxRM-RF(mse)
            max_risks[:, 3],  # MaxRM-RF(mse)
            max_risks[:, 4],  # Oracle MaxRM-RF(reg)
            max_risks[:, 5],  # MaxRM-RF(reg)
        ],
        tick_labels=[
            "Oracle\nRF",
            "RF",
            "Oracle\nMaxRM-RF(mse)",
            "MaxRM-RF(mse)",
            "Oracle\nMaxRM-RF(reg)",
            "MaxRM-RF(reg)",
        ],
        patch_artist=True,
    )
    ax.tick_params(axis="x", labelsize=7)
    # Pair colors: RF (blue), MSE (orange), Regret (green).
    # Oracle is lighter shade.
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

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
    ]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
    ax.set_ylabel("Maximum MSE across environments")
    ax.set_title("Comparison of methods with different noise levels")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"boxplot_diff_noise_changeXdistr{str(CHANGE_X_DISTR)}_leaf{MIN_SAMPLES_LEAF}_reps{N_SIM}_p{NUM_COVARIATES}.png",
        )
    )
    plt.close(fig)
