import copy
from nldg.utils import *
from adaXT.random_forest import RandomForest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

SIGMA_EPS_BASE = 0.1
KERNEL = ConstantKernel(1.0) * RBF(length_scale=0.5)
SEED = 0
RNG = np.random.default_rng(SEED)
N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 50
E = 3  # number of training environments
N_TRAIN_PER_ENV = [1500, 1500, 1500]
N_TEST = 1500
N_SIM = 100
GRID_SIZE = 15
Q1_VALS = np.linspace(0, 1, GRID_SIZE)
Q2_VALS = np.linspace(0, 1, GRID_SIZE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SIM_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(SIM_DIR, exist_ok=True)
OUT_DIR = os.path.join(SIM_DIR, "sim_mse_degeneration")
os.makedirs(OUT_DIR, exist_ok=True)


def sample_gp_function(x_grid):
    K = KERNEL(x_grid, x_grid)
    f_vals = RNG.multivariate_normal(np.zeros(len(x_grid)), K)
    return lambda x: np.interp(x.ravel(), x_grid.ravel(), f_vals)


def make_dataset(f, env_id, n_samples):
    X = RNG.uniform(-1, 1, size=(n_samples, 1))
    y_clean = f(X)
    env_lab = np.full(n_samples, env_id)
    return X, y_clean, env_lab


def generate_data():
    global f_env
    train_sets = [
        make_dataset(f_env[e], env_id=e, n_samples=N_TRAIN_PER_ENV[e])
        for e in range(E)
    ]

    X_tr = np.vstack([ts[0] for ts in train_sets])
    y_tr_clean = np.hstack([ts[1] for ts in train_sets])
    env_label = np.hstack([ts[2] for ts in train_sets])

    return X_tr, y_tr_clean, env_label


def plot_tricontour(diff_map, metric):
    q1_grid, q2_grid, diff_grid = [], [], []
    for (q1, q2), diffs in diff_map.items():
        q1_grid.append(q1)
        q2_grid.append(q2)
        diff_grid.append(np.mean(diffs))

    plt.figure(figsize=(8, 7))
    sc = plt.tricontourf(q1_grid, q2_grid, diff_grid, levels=30, cmap="Blues")

    cbar = plt.colorbar(sc, pad=0.02, aspect=30)
    if metric == "mse":
        lab = "MSE"
    elif metric == "negrew":
        lab = "NRW"
    else:
        lab = "Reg"
    cbar.set_label(rf"$\overline{{D}}_{{e^\prime}}^{{{lab}}}$", fontsize=14)
    # if metric == "mse":
    #     lab = "MSE"
    # elif metric == "negrew":
    #     lab = "Negative reward"
    # else:
    #     lab = "Regret"
    # cbar.set_label(f"Average Generalization Gap ({lab})", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel("$q_1$", fontsize=12)
    plt.ylabel("$q_2$", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot([0, 1, 0], [0, 0, 1], color="black", lw=1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"{metric}_diff_tricontour_DEGENERATION.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # Generate training data (without noise)
    x_grid = np.linspace(-1, 1, 1000).reshape(-1, 1)
    f_env = [sample_gp_function(x_grid) for _ in range(E)]
    X_tr, y_tr_clean, env_label = generate_data()

    # Add noise
    # Let the third environment be the noisiest one
    # Compute, for each environment e != 2, E[(f^e(X)-f^{2}(X))^2]
    k = 2
    max_delta = 0
    for e in range(E):
        if e != k:
            mask = env_label == e
            delta = np.mean((y_tr_clean[mask] - f_env[k](X_tr[mask])) ** 2)
            max_delta = max(max_delta, delta)

    # Therefore, we need to choose sigma_2^2 >= max_delta + sigma^2
    y_tr = copy.deepcopy(y_tr_clean)
    sigma_max = np.sqrt(max_delta + SIGMA_EPS_BASE**2)
    for e in range(E):
        mask = env_label == e
        if e != k:
            y_tr[mask] += RNG.normal(
                0, SIGMA_EPS_BASE, size=N_TRAIN_PER_ENV[e]
            )
        else:
            y_tr[mask] += RNG.normal(0, sigma_max, size=N_TRAIN_PER_ENV[e])

    # Model fitting
    # MSE
    rf_mse = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf_mse.fit(X_tr, y_tr)
    rf_mse.modify_predictions_trees(env_label)

    # Negative Reward
    rf_negrew = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf_negrew.fit(X_tr, y_tr)
    rf_negrew.modify_predictions_trees(env_label, method="reward")

    # Regret
    sols_erm = np.zeros(env_label.shape[0])
    sols_erm_trees = np.zeros((N_ESTIMATORS, env_label.shape[0]))
    for env in np.unique(env_label):
        mask = env_label == env
        X_e = X_tr[mask]
        Y_e = y_tr[mask]
        rf_e = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf_e.fit(X_e, Y_e)
        fitted_e = rf_e.predict(X_e)
        sols_erm[mask] = fitted_e
        for i in range(N_ESTIMATORS):
            fitted_e_tree = rf_e.trees[i].predict(X_e)
            sols_erm_trees[i, mask] = fitted_e_tree
    rf_regret = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
    )
    rf_regret.fit(X_tr, y_tr)
    rf_regret.modify_predictions_trees(
        env_label,
        method="regret",
        sols_erm=sols_erm,
        sols_erm_trees=sols_erm_trees,
    )

    # Comparison
    max_mse_tr_list, max_negrew_tr_list, max_regret_tr_list = [], [], []
    max_mse_te_list, max_negrew_te_list, max_regret_te_list = [], [], []
    mse_diff_map, negrew_diff_map, regret_diff_map = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for i in tqdm(range(N_SIM)):
        X_tr_new, y_tr_clean_new, env_label = generate_data()

        y_tr_new = copy.deepcopy(y_tr_clean_new)
        for e in range(E):
            mask = env_label == e
            if e != k:
                y_tr_new[mask] += RNG.normal(
                    0, SIGMA_EPS_BASE, size=N_TRAIN_PER_ENV[e]
                )
            else:
                y_tr_new[mask] += RNG.normal(
                    0, sigma_max, size=N_TRAIN_PER_ENV[e]
                )

        # MSE
        fitted_mse = rf_mse.predict(X_tr_new)
        max_mse_tr = max_mse(y_tr_new, fitted_mse, env_label)
        max_mse_tr_list.append(max_mse_tr)

        # Negative Reward
        fitted_negrew = rf_negrew.predict(X_tr_new)
        max_negrew_tr = -min_reward(y_tr_new, fitted_negrew, env_label)
        max_negrew_tr_list.append(max_negrew_tr)

        # Regret
        sols_erm_new = np.zeros(env_label.shape[0])
        for env in np.unique(env_label):
            mask = env_label == env
            # X_e = X_tr_new[mask]
            # Y_e = y_tr_new[mask]
            # rf_e = RandomForest(
            #     "Regression",
            #     n_estimators=N_ESTIMATORS,
            #     min_samples_leaf=MIN_SAMPLES_LEAF,
            #     seed=SEED,
            # )
            # rf_e.fit(X_e, Y_e)
            # fitted_e = rf_e.predict(X_e)
            # sols_erm_new[mask] = fitted_e
            sols_erm_new[mask] = y_tr_clean_new[mask]
        fitted_regret = rf_regret.predict(X_tr_new)
        max_regret_tr = max_regret(
            y_tr_new, fitted_regret, sols_erm_new, env_label
        )
        max_regret_tr_list.append(max_regret_tr)

        # Test environment
        X_te = RNG.uniform(-1, 1, size=(N_TEST, 1))
        eps_te = RNG.normal(0, SIGMA_EPS_BASE, size=N_TEST)

        preds_mse = rf_mse.predict(X_te)
        preds_negrew = rf_negrew.predict(X_te)
        preds_regret = rf_regret.predict(X_te)

        max_mse_te, max_negrew_te, max_regret_te = -np.inf, -np.inf, -np.inf
        for q1, q2 in product(Q1_VALS, Q2_VALS):
            if q1 + q2 > 1:
                continue
            q3 = 1 - q1 - q2
            q = [q1, q2, q3]
            f_te = lambda x: sum(q[e] * f_env[e](x) for e in range(E))
            y_te_clean = f_te(X_te)
            y_te = y_te_clean + eps_te
            # y_te = y_te - np.mean(y_te)

            # MSE
            mse_te = mean_squared_error(y_te, preds_mse)
            max_mse_te = max(max_mse_te, mse_te)
            mse_diff = mse_te - max_mse_tr
            mse_diff_map[(q1, q2)].append(mse_diff)

            # Negative reward
            negrew_te = mean_squared_error(y_te, preds_negrew) - np.mean(
                y_te**2
            )
            max_negrew_te = max(max_negrew_te, negrew_te)
            negrew_diff = negrew_te - max_negrew_tr
            negrew_diff_map[(q1, q2)].append(negrew_diff)

            # Regret
            # rf_regret_te = RandomForest(
            #     "Regression",
            #     n_estimators=N_ESTIMATORS,
            #     min_samples_leaf=MIN_SAMPLES_LEAF,
            #     seed=SEED,
            # )
            # rf_regret_te.fit(X_te, y_te)
            # sols_erm_te = rf_regret_te.predict(X_te)
            sols_erm_te = y_te_clean
            regret_te = mean_squared_error(
                y_te, preds_regret
            ) - mean_squared_error(y_te, sols_erm_te)
            max_regret_te = max(max_regret_te, regret_te)
            regret_diff = regret_te - max_regret_tr
            regret_diff_map[(q1, q2)].append(regret_diff)

        max_mse_te_list.append(max_mse_te)
        max_negrew_te_list.append(max_negrew_te)
        max_regret_te_list.append(max_regret_te)

    ret_mse = stats.ttest_ind(
        max_mse_te_list, max_mse_tr_list, equal_var=False
    )
    ci_mse = ret_mse.confidence_interval(confidence_level=0.95)

    ret_negrew = stats.ttest_ind(
        max_negrew_te_list, max_negrew_tr_list, equal_var=False
    )
    ci_negrew = ret_negrew.confidence_interval(confidence_level=0.95)

    ret_regret = stats.ttest_ind(
        max_regret_te_list, max_regret_tr_list, equal_var=False
    )
    ci_regret = ret_regret.confidence_interval(confidence_level=0.95)

    output_path = os.path.join(OUT_DIR, "results_DEGENERATION.txt")
    with open(output_path, "w") as f:
        f.write("MSE\n")
        f.write(f"Statistic: {ret_mse.statistic:.4f}\n")
        f.write(f"p-value: {ret_mse.pvalue:.4g}\n")
        f.write(
            f"(95%) Confidence Interval: [{ci_mse.low:.4f}, {ci_mse.high:.4f}]\n"
        )

        f.write("\nNegative Reward\n")
        f.write(f"Statistic: {ret_negrew.statistic:.4f}\n")
        f.write(f"p-value: {ret_negrew.pvalue:.4g}\n")
        f.write(
            f"(95%) Confidence Interval: [{ci_negrew.low:.4f}, {ci_negrew.high:.4f}]\n"
        )

        f.write("\nRegret\n")
        f.write(f"Statistic: {ret_regret.statistic:.4f}\n")
        f.write(f"p-value: {ret_regret.pvalue:.4g}\n")
        f.write(
            f"(95%) Confidence Interval: [{ci_regret.low:.4f}, {ci_regret.high:.4f}]\n"
        )

    plot_tricontour(mse_diff_map, "mse")
    plot_tricontour(negrew_diff_map, "negrew")
    plot_tricontour(regret_diff_map, "regret")

    print(f"Saved results to {OUT_DIR}")
