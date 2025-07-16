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
OUT_DIR = os.path.join(SIM_DIR, "sim_gen_gap")
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
    # cbar.set_label(
    #     r"$\max_{e^\prime\in\mathcal{E}_{\text{te}}^\prime}\hat{R}_{e^\prime}^{\text{MSE}}(\bm{c}^*) - \hat{R}^{\text{MSE}}(\bm{c}^*)$",
    #     fontsize=12
    # )
    if metric == "mse":
        lab = "MSE"
    elif metric == "negrew":
        lab = "Negative reward"
    else:
        lab = "Regret"
    cbar.set_label(f"Average Generalization Gap ({lab})", fontsize=12)
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

    # Comparison
    max_mse_tr_list = []
    max_mse_te_list = []
    mse_diff_map = defaultdict(list)
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

        # Test environment
        X_te = RNG.uniform(-1, 1, size=(N_TEST, 1))
        eps_te = RNG.normal(0, SIGMA_EPS_BASE, size=N_TEST)

        preds_mse = rf_mse.predict(X_te)

        max_mse_te = -np.inf
        for q1, q2 in product(Q1_VALS, Q2_VALS):
            if q1 + q2 > 1:
                continue
            q3 = 1 - q1 - q2
            q = [q1, q2, q3]
            f_te = lambda x: sum(q[e] * f_env[e](x) for e in range(E))
            y_te = f_te(X_te) + eps_te
            # y_te = y_te - np.mean(y_te)

            # MSE
            mse_te = mean_squared_error(y_te, preds_mse)
            max_mse_te = max(max_mse_te, mse_te)
            mse_diff = mse_te - max_mse_tr
            mse_diff_map[(q1, q2)].append(mse_diff)

        max_mse_te_list.append(max_mse_te)

    ret_mse = stats.ttest_ind(
        max_mse_te_list, max_mse_tr_list, equal_var=False
    )
    ci_mse = ret_mse.confidence_interval(confidence_level=0.95)

    output_path = os.path.join(OUT_DIR, "mse_results_DEGENERATION.txt")
    with open(output_path, "w") as f:
        f.write("MSE\n")
        f.write(f"Statistic: {ret_mse.statistic:.4f}\n")
        f.write(f"p-value: {ret_mse.pvalue:.4g}\n")
        f.write(
            f"(95%) Confidence Interval: [{ci_mse.low:.4f}, {ci_mse.high:.4f}]\n"
        )

    plot_tricontour(mse_diff_map, "mse")

    print(f"Saved results to {OUT_DIR}")
