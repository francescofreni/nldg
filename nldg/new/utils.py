import numpy as np
import pandas as pd
import random
import torch
from scipy.stats import ortho_group
from scipy.linalg import block_diag


# ===============
# DATA GENERATION
# ===============
def gen_data(
    n_train: int = 500,
    n_test: int = 100,
    n_envs: int = 5,
    random_state: int = 42,
    inhull: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates train and test data for a nonlinear setting.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        n_envs: Number of training environments (groups).
        random_state: Random seed.
        inhull: If True, test function is in the convex hull of training functions.
            If False, test function is outside the convex hull.

    Returns:
        A tuple containing:
        - df_train: DataFrame with training data (X, y, env).
        - df_test: DataFrame with test data (X, y, env).
    """
    rng = np.random.default_rng(random_state)
    n_e = n_train // n_envs

    env_coeffs = [rng.uniform(-3, 3, size=3) for _ in range(n_envs)]

    train_data = []
    for env_id, coeff in enumerate(env_coeffs):
        x_env = rng.normal(0, 1, size=n_e)
        y_env = (
            coeff[0] * x_env**2
            + coeff[1] * x_env
            + coeff[2]
            + rng.normal(0, 0.5, size=n_e)
        )
        train_data.append(pd.DataFrame({"X": x_env, "Y": y_env, "E": env_id}))

    df_train = pd.concat(train_data, ignore_index=True)

    if inhull:
        weights = rng.dirichlet(alpha=np.ones(n_envs))
        test_coeff = np.sum(np.array(env_coeffs).T * weights, axis=1)
    else:
        test_coeff = rng.uniform(-5.0, -4.0, size=3)

    x_test = rng.normal(0, 1, size=n_test)
    y_test = (
        test_coeff[0] * x_test**2
        + test_coeff[1] * x_test
        + test_coeff[2]
        + rng.normal(0, 0.5, size=n_test)
    )

    df_test = pd.DataFrame({"X": x_test, "Y": y_test, "E": -1})

    return df_train, df_test


def gen_data_v2(
    n_train: int = 500,
    n_test: int = 100,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates train and test data for two environments.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        random_state: Random seed.

    Returns:
        A tuple containing:
        - df_train: DataFrame with training data (X, Y, E).
        - df_test: DataFrame with test data (X, Y, E).
    """
    rng = np.random.default_rng(random_state)

    n_e1 = n_train // 2
    n_e1_left = int(0.9 * n_e1)
    n_e1_right = n_e1 - n_e1_left
    n_e2 = n_train - n_e1
    n_e2_right = int(0.9 * n_e2)
    n_e2_left = n_e1 - n_e2_right

    x_e1_left = rng.normal(0, 1, size=n_e1_left)
    x_e1_right = rng.normal(4, 1, size=n_e1_right)
    x_e2_left = rng.normal(0, 1, size=n_e2_left)
    x_e2_right = rng.normal(4, 1, size=n_e2_right)

    noise_std = 0.2
    y_e1_left = (
        -np.sin(x_e1_left) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_e1_left)
    )
    y_e1_right = np.sin(x_e1_right + np.pi) ** 2 + noise_std * rng.normal(
        1.5, 1, size=n_e1_right
    )
    y_e2_left = np.sin(x_e2_left) ** 2 + noise_std * rng.normal(
        0, 1, size=n_e2_left
    )
    y_e2_right = (
        -np.sin(x_e2_right + np.pi) ** 2
        + 3
        + noise_std * rng.normal(1.5, 1, size=n_e2_right)
    )

    df_train_e1 = pd.DataFrame(
        {
            "X": np.concatenate([x_e1_left, x_e1_right]),
            "Y": np.concatenate([y_e1_left, y_e1_right]),
            "E": 0,
        }
    )

    df_train_e2 = pd.DataFrame(
        {
            "X": np.concatenate([x_e2_left, x_e2_right]),
            "Y": np.concatenate([y_e2_left, y_e2_right]),
            "E": 1,
        }
    )

    df_train = pd.concat([df_train_e1, df_train_e2], ignore_index=True)

    n_test_left = n_test // 2
    n_test_right = n_test - n_test_left
    x_test_left = rng.normal(0, 1, size=n_test_left)
    x_test_right = rng.normal(3.5, 1, size=n_test_left)
    y_test_left = (
        -np.sin(x_test_left) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_test_left)
    )
    y_test_right = (
        -np.sin(x_test_right + np.pi) ** 2
        + 3
        + noise_std * rng.normal(1.5, 1, size=n_test_right)
    )

    df_test = pd.DataFrame(
        {
            "X": np.concatenate([x_test_left, x_test_right]),
            "Y": np.concatenate([y_test_left, y_test_right]),
            "E": -1,
        }
    )

    return df_train, df_test


def gen_data_v3(
    n_train: int = 500,
    n_test: int = 100,
    random_state: int = 0,
    setting: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates train data from three environments.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        random_state: Random seed.
        setting: Data setting. Current accepted values are 1 or 2

    Returns:
        A tuple containing:
        - df_train: DataFrame with training data (X1, X2, Y, E).
        - df_test: DataFrame with test data (X1, X2, Y, E).
    """
    possible_settings = [1, 2]
    if setting not in [1, 2]:
        raise ValueError(f"setting must be in {possible_settings}.")
    rng = np.random.default_rng(random_state)
    Sigma = np.array([[1, 0.0], [0.0, 1]])
    sigma = 0.3
    n_e = n_train // 3

    def generate_environment(env_id, n):
        X = rng.multivariate_normal(mean=[0, 0], cov=Sigma, size=n)
        eps = sigma * rng.normal(0, 1, size=n)

        if env_id == 0:
            Y = 5 * np.sin(X[:, 0]) + 2 * X[:, 1] + eps  # results
            # Y = 5 * np.sin(X[:, 0]) + np.abs(X[:, 1]) + eps  # results_3
            # Y = 5 * np.sin(X[:, 0]) + (X[:, 1] + 1) ** 2 + eps  # results_4
        elif env_id == 1:
            Y = 5 * np.sin(X[:, 0]) - 2 * X[:, 1] + eps  # results
            # Y = 5 * np.sin(X[:, 0]) - np.exp(X[:, 1] / 3) + eps  # results_3
            # Y = 5 * np.sin(X[:, 0]) + np.cos(X[:, 1]) + 1 + eps  # results_4
        elif env_id == 2:
            Y = 5 * np.sin(X[:, 0]) + X[:, 1] ** 2 + eps  # results
            # Y = 5 * np.sin(X[:, 0]) + np.cos(X[:, 1]) + eps  # results_3
            # Y = 5 * np.sin(X[:, 0]) + 6 * np.sin(X[:, 1]) + eps  # results_4
        else:
            raise ValueError("Invalid environment ID")

        return pd.DataFrame(
            {"X1": X[:, 0], "X2": X[:, 1], "Y": Y, "E": env_id}
        )

    df_train = pd.concat(
        [generate_environment(env_id, n_e) for env_id in range(3)],
        ignore_index=True,
    )

    X = rng.multivariate_normal(mean=[0, 0], cov=Sigma, size=n_test)
    eps = rng.normal(0, 1, size=n_test)
    if setting == 1:
        Y = 5 * np.sin(X[:, 0]) + eps
    else:
        Y = 5 * np.sin(X[:, 0]) + 3 * np.cos(X[:, 0]) + eps
    df_test = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Y": Y, "E": -1})

    return df_train, df_test


def gen_data_isd(
    n_train: int = 1500,
    n_test: int = 500,
    p: int = 2,
    block_sizes: list = [1, 1],
    random_state: int = 0,
    setting: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates train data from three environments.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        p: Number of variables.
        random_state: Random seed.
        setting: Data setting. Current accepted values are 1 or 2

    Returns:
        A tuple containing:
        - df_train: DataFrame with training data (X1, X2, Y, E).
        - df_test: DataFrame with test data (X1, X2, Y, E).
    """
    sigma = 0.5
    rng = np.random.default_rng(random_state)
    rng_sigma = np.random.default_rng(42)
    OM = ortho_group.rvs(dim=p, random_state=rng)
    n_envs = 3
    n_e = n_train // n_envs
    eps = sigma * rng.normal(0, 1, size=n_train)

    X = np.zeros((n_train, p))
    E = np.zeros((n_train,))
    for e in range(n_envs):
        A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
        Sigma_e = OM.T @ (A @ A.T + 0.0 * np.eye(p)) @ OM
        X_e = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma_e, size=n_e)
        X[(e * n_e) : ((e + 1) * n_e)] = X_e
        E[(e * n_e) : ((e + 1) * n_e)] = e

    X_rot = X @ OM.T
    for e in range(n_envs):
        if e == 0:
            Y = 5 * np.sin(X_rot[:, 0]) + 2 * X_rot[:, 1] + eps
        elif e == 1:
            Y = 5 * np.sin(X_rot[:, 0]) - 2 * X_rot[:, 1] + eps
        else:
            Y = 5 * np.sin(X_rot[:, 0]) + X_rot[:, 1] ** 2 + eps

    df_train = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Y": Y, "E": E})

    eps = sigma * rng.normal(0, 1, size=n_test)
    A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
    Sigma_e = OM.T @ (A @ A.T + 0.0 * np.eye(p)) @ OM
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma_e, size=n_test)
    X_rot = X @ OM.T
    if setting == 1:
        Y = 5 * np.sin(X_rot[:, 0]) + eps
    else:
        Y = 5 * np.sin(X_rot[:, 0]) + 3 * np.cos(X_rot[:, 0]) + eps
    df_test = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Y": Y, "E": -1})

    return df_train, df_test


# ========================
# NEURAL NETWORK UTILITIES
# ========================
def set_all_seeds(seed: int):
    """
    Set all possible seeds to ensure reproducibility and to avoid randomness
    involved in GPU computations.

    Args:
        seed (int): Seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======
# OTHERS
# ======
def max_mse(
    Ytrue: np.ndarray,
    Ypred: np.ndarray,
    Env: np.ndarray,
    verbose: bool = False,
) -> float:
    """
    Compute the maximum mean squared error (MSE) across environments.

    Args:
        Ytrue (array): True target values.
        Ypred (array): Predicted target values.
        Env (array): Environment values.
        verbose: Whether to print the MSE for each environment.

    Returns:
        maxmse (float): Maximum mean squared error.
    """
    maxmse = 0.0
    for env in np.unique(Env):
        Ytrue_e = Ytrue[Env == env]
        Ypred_e = Ypred[Env == env]
        mse = np.mean((Ytrue_e - Ypred_e) ** 2)
        if verbose:
            print(f"Environment {env} MSE: {mse}")
        maxmse = max(maxmse, mse)
    return maxmse
