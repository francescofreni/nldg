import os
from nldg.utils import *
from adaXT.random_forest import RandomForest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm
from scipy import stats
from collections import defaultdict
import matplotlib as mpl

SIGMA_EPS = 0.1  # assume homoskedasticity
KERNEL = ConstantKernel(1.0) * RBF(length_scale=0.5)
SEED = 42
RNG = np.random.default_rng(SEED)
N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 20
E = 3  # number of training environments
N_TRAIN_PER_ENV = [1000, 1000, 1000]
N_TEST = 1000
N_SIM = 100
GRID_SIZE = 100
Q1_VALS = np.linspace(0, 1, GRID_SIZE)
Q2_VALS = np.linspace(0, 1, GRID_SIZE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(OUT_DIR, exist_ok=True)

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['text.latex.preamble'] = r'''
# \usepackage{amsmath}
# \usepackage{amssymb}
# \usepackage{bm}
# '''


def sample_gp_function(x_grid):
    K = KERNEL(x_grid, x_grid)
    f_vals = RNG.multivariate_normal(np.zeros(len(x_grid)), K)
    return lambda x: np.interp(x.ravel(), x_grid.ravel(), f_vals)


def make_dataset(f, env_id, n_samples):
    X = RNG.uniform(-1, 1, size=(n_samples, 1))
    eps = RNG.normal(0, SIGMA_EPS, size=n_samples)
    y = f(X) + eps
    env_lab = np.full(n_samples, env_id)
    return X, y, env_lab


def generate_data():
    global f_env
    train_sets = [
        make_dataset(f_env[e], env_id=e, n_samples=N_TRAIN_PER_ENV[e])
        for e in range(E)
    ]

    X_tr = np.vstack([ts[0] for ts in train_sets])
    y_tr = np.hstack([ts[1] for ts in train_sets])
    env_label = np.hstack([ts[2] for ts in train_sets])

    return X_tr, y_tr, env_label


def plot_tricontour(diff_map):
    q1_grid, q2_grid, diff_grid = [], [], []
    for (q1, q2), diffs in diff_map.items():
        q1_grid.append(q1)
        q2_grid.append(q2)
        diff_grid.append(np.mean(diffs))

    plt.figure(figsize=(8, 7))
    sc = plt.tricontourf(q1_grid, q2_grid, diff_grid, levels=100, cmap="Blues")

    cbar = plt.colorbar(sc, pad=0.02, aspect=30)
    # cbar.set_label(
    #     r"$\max_{e^\prime\in\mathcal{E}_{\text{te}}^\prime}\hat{R}_{e^\prime}^{\text{MSE}}(\bm{c}^*) - \hat{R}^{\text{MSE}}(\bm{c}^*)$",
    #     fontsize=12
    # )
    cbar.set_label("Average Generalization Gap", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel("$q_1$", fontsize=12)
    plt.ylabel("$q_2$", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot([0, 1, 0], [0, 0, 1], color="black", lw=1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "mse_diff_tricontour.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    x_grid = np.linspace(-1, 1, 1000).reshape(-1, 1)
    f_env = [sample_gp_function(x_grid) for _ in range(E)]
    max_mse_tr_list = []
    max_mse_te_list = []
    mse_diff_map = defaultdict(list)
    for i in tqdm(range(N_SIM)):
        X_tr, y_tr, env_label = generate_data()

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
        )
        rf.fit(X_tr, y_tr)

        rf.modify_predictions_trees(env_label)
        fitted_posthoc = rf.predict(X_tr)

        max_mse_tr = max_mse(y_tr, fitted_posthoc, env_label)
        max_mse_tr_list.append(max_mse_tr)

        X_te = RNG.uniform(-1, 1, size=(N_TEST, 1))
        eps_te = RNG.normal(0, SIGMA_EPS, size=N_TEST)
        max_mse_te = -np.inf
        for q1, q2 in product(Q1_VALS, Q2_VALS):
            if q1 + q2 > 1:
                continue
            q3 = 1 - q1 - q2
            q = [q1, q2, q3]
            f_te = lambda x: sum(q[e] * f_env[e](x) for e in range(E))
            y_te = f_te(X_te) + eps_te
            preds_posthoc = rf.predict(X_te)
            mse_te = mean_squared_error(y_te, preds_posthoc)
            max_mse_te = max(max_mse_te, mse_te)
            mse_diff = mse_te - max_mse_tr
            mse_diff_map[(q1, q2)].append(mse_diff)
        max_mse_te_list.append(max_mse_te)

        ret = stats.ttest_ind(
            max_mse_te_list, max_mse_tr_list, equal_var=False
        )
        ci = ret.confidence_interval(confidence_level=0.95)

        output_path = os.path.join(OUT_DIR, "mse_results.txt")
        with open(output_path, "w") as f:
            f.write(f"Statistic: {ret.statistic:.4f}\n")
            f.write(f"p-value: {ret.pvalue:.4g}\n")
            f.write(
                f"(95%) Confidence Interval: [{ci.low:.4f}, {ci.high:.4f}]\n"
            )

        plot_tricontour(mse_diff_map)
