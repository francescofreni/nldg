import numpy as np
import pandas as pd


# ===============
# DATA GENERATION
# ===============
def gen_data(
    n_train: int = 500,
    n_test: int = 100,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates train data from three environments.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        random_state: Random seed.

    Returns:
        A tuple containing:
        - df_train: DataFrame with training data.
        - df_test_1: DataFrame with test data.
        - df_test_2 DataFrame with different test data.
    """
    rng = np.random.default_rng(random_state)
    p = 2
    Sigma = np.eye(p) * 3.5
    sigma = 0.3
    n_e = n_train // 3

    def generate_environment(env_id, n):
        m = np.zeros(p)
        X = rng.multivariate_normal(mean=m, cov=Sigma, size=n)
        eps = sigma * rng.normal(0, 1, size=n)

        if env_id == 0:
            Y = 2 * np.sin(X[:, 0]) + 2 * X[:, 1] + eps
        elif env_id == 1:
            Y = 2 * np.sin(X[:, 0]) + 3 * X[:, 1] + eps
        elif env_id == 2:
            Y = 2 * np.sin(X[:, 0]) - 3 * X[:, 1] + eps
        else:
            raise ValueError("Invalid environment ID")

        data = {f"X{i+1}": X[:, i] for i in range(p)}
        data.update({"Y": Y, "E": env_id})

        return pd.DataFrame(data)

    df_train = pd.concat(
        [generate_environment(env_id, n_e) for env_id in range(3)],
        ignore_index=True,
    )

    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n_test)
    eps = rng.normal(0, 1, size=n_test)
    Y1 = 2 * np.sin(X[:, 0]) + eps
    Y2 = 2 * np.sin(X[:, 0]) + 3 * np.cos(X[:, 1]) + eps
    df_test_1 = pd.DataFrame({f"X{i+1}": X[:, i] for i in range(p)})
    df_test_1["Y"] = Y1
    df_test_1["E"] = -1

    df_test_2 = pd.DataFrame({f"X{i + 1}": X[:, i] for i in range(p)})
    df_test_2["Y"] = Y2
    df_test_2["E"] = -1

    return df_train, df_test_1, df_test_2


def gen_data_v2(
    n: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    n_e1 = n // 2
    n_e1_left = int(0.9 * n_e1)
    n_e1_right = n_e1 - n_e1_left
    n_e2 = n - n_e1
    n_e2_right = int(0.9 * n_e2)
    n_e2_left = n_e1 - n_e2_right

    x_e1_left = rng.normal(np.pi, 1, size=n_e1_left)
    x_e1_right = rng.normal(3 * np.pi, 1, size=n_e1_right)
    x_e2_left = rng.normal(np.pi, 1, size=n_e2_left)
    x_e2_right = rng.normal(3 * np.pi, 1, size=n_e2_right)

    noise_std = 0.2
    y_e1_left = (
        np.sin(x_e1_left / 2) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_e1_left)
    )
    y_e1_right = (
        -np.sin(x_e1_right / 2) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_e1_right)
    )
    y_e2_left = (
        -np.sin(x_e2_left / 2) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_e2_left)
    )
    y_e2_right = (
        np.sin(x_e2_right / 2) ** 2
        + 3
        + noise_std * rng.normal(0, 1, size=n_e2_right)
    )

    df_e1 = pd.DataFrame(
        {
            "X": np.concatenate([x_e1_left, x_e1_right]),
            "Y": np.concatenate([y_e1_left, y_e1_right]),
            "E": 0,
        }
    )

    df_e2 = pd.DataFrame(
        {
            "X": np.concatenate([x_e2_left, x_e2_right]),
            "Y": np.concatenate([y_e2_left, y_e2_right]),
            "E": 1,
        }
    )

    df = pd.concat([df_e1, df_e2], ignore_index=True)

    return df


def gen_data_v3(
    n: int = 500, random_state: int = 0, setting: int = 1
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    sigma = 1
    n_e = n // 3

    perc = 0.5 if setting == 1 else 0.9
    n_e1_right = int(perc * n_e)
    n_e1_left = n_e - n_e1_right
    x_e1_left = sigma * rng.normal(-2, 1, size=n_e1_left)
    x_e1_right = sigma * rng.normal(2, 1, size=n_e1_right)
    x_e1 = np.concatenate([x_e1_left, x_e1_right])

    n_e2_left = int(perc * n_e)
    n_e2_right = n_e - n_e2_left
    x_e2_left = sigma * rng.normal(-2, 1, size=n_e2_left)
    x_e2_right = sigma * rng.normal(2, 1, size=n_e2_right)
    x_e2 = np.concatenate([x_e2_left, x_e2_right])

    n_e3_right = int(perc * n_e)
    n_e3_left = n_e - n_e3_right
    x_e3_left = sigma * rng.normal(-2, 1, size=n_e3_left)
    x_e3_right = sigma * rng.normal(2, 1, size=n_e3_right)
    x_e3 = np.concatenate([x_e3_left, x_e3_right])

    noise_std = 1
    y_e1 = 3 * x_e1 + noise_std * rng.normal(0, 1, size=n_e)
    y_e2 = -3 * x_e2 + noise_std * rng.normal(0, 1, size=n_e)
    y_e3 = 2 * x_e3 + noise_std * rng.normal(0, 1, size=n_e)
    df_e1 = pd.DataFrame({"X": x_e1, "Y": y_e1, "E": 0})
    df_e2 = pd.DataFrame({"X": x_e2, "Y": y_e2, "E": 1})
    df_e3 = pd.DataFrame({"X": x_e3, "Y": y_e3, "E": 2})

    df = pd.concat([df_e1, df_e2, df_e3], ignore_index=True)

    return df


def gen_data_v4(
    n_easy: int = 300, n_hard: int = 300, random_state: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    X_easy = rng.uniform(-5, 5, size=n_easy)
    noise_easy = rng.normal(0, 0.5, size=n_easy)
    y_easy = 3 * X_easy + 2 + noise_easy
    df_easy = pd.DataFrame({"X": X_easy, "Y": y_easy, "E": 0})

    X_hard = rng.uniform(-5, 5, size=n_hard)
    noise_hard = rng.normal(0, 2.0, size=n_hard)
    y_hard = 0.5 * X_hard + 5 + 0.3 * (X_hard**2) + noise_hard
    df_hard = pd.DataFrame({"X": X_hard, "Y": y_hard, "E": 1})

    df_all = pd.concat([df_easy, df_hard], ignore_index=True)
    return df_all


def gen_data_v5(
    n_samples: int = 500,
    adv_fraction: float = 0.1,
    noise_var_env2=10.0,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    X = rng.uniform(-5, 5, size=(n_samples, 1))

    y = 2.0 * X.squeeze() + 1.0 + rng.normal(0, 1.0, size=n_samples)

    n_adv = int(adv_fraction * n_samples)
    adv_indices = rng.choice(n_samples, n_adv, replace=False)

    # Inject adversarial noise: high-variance noise + slight non-linear distortion
    y[adv_indices] += rng.normal(0, noise_var_env2, size=n_adv) + 0.5 * (
        X[adv_indices].squeeze() ** 2
    )  # + 10

    env = np.zeros(n_samples, dtype=int)
    env[adv_indices] = 1

    df = pd.DataFrame({"X": X.squeeze(), "Y": y, "E": env})

    return df


# ==========
# EVALUATION
# ==========
def max_mse(
    Ytrue: np.ndarray,
    Ypred: np.ndarray,
    Env: np.ndarray,
    verbose: bool = False,
    ret_ind: bool = False,
) -> float | tuple[list, float]:
    """
    Compute the maximum mean squared error (MSE) across environments.

    Args:
        Ytrue (array): True target values.
        Ypred (array): Predicted target values.
        Env (array): Environment values.
        verbose (bool): Whether to print the MSE for each environment.
        ret_ind (bool): Whether to return also the MSE for each environment.

    Returns:
        if ret_ind:
            mse_envs (list): MSE for each environment.
        maxmse (float): Maximum mean squared error.
    """
    maxmse = 0.0
    mse_envs = []
    for env in np.unique(Env):
        Ytrue_e = Ytrue[Env == env]
        Ypred_e = Ypred[Env == env]
        mse = np.mean((Ytrue_e - Ypred_e) ** 2)
        mse_envs.append(mse)
        if verbose:
            print(f"Environment {env} MSE: {mse}")
        maxmse = max(maxmse, mse)
    if ret_ind:
        return mse_envs, maxmse
    return maxmse


def min_xplvar(
    Ytrue: np.ndarray,
    Ypred: np.ndarray,
    Env: np.ndarray,
    verbose: bool = False,
    ret_ind: bool = False,
) -> float | tuple[list, float]:
    """
    Compute the minimum explained variance across environments.

    For each environment, the explained variance is computed as:
        EV = mean(Ytrue^2) - mean((Ytrue - Ypred)^2)
    This function returns the minimum EV across all environments.

    Args:
        Ytrue (np.ndarray): True target values.
        Ypred (np.ndarray): Predicted target values.
        Env (np.ndarray): Environment labels for each sample.
        verbose (bool): Whether to print the explained variance for each environment.
        ret_ind (bool): Whether to return also the explained variance for each environment.

    Returns:
        if ret_ind:
            xplvar_envs (list): Explained variance for each environment.
        min_ev (float): Minimum explained variance across environments.
    """
    min_ev = float("inf")
    xplvar_envs = []
    for env in np.unique(Env):
        Ytrue_e = Ytrue[Env == env]
        Ypred_e = Ypred[Env == env]
        ev = np.mean(Ytrue_e**2) - np.mean((Ytrue_e - Ypred_e) ** 2)
        xplvar_envs.append(ev)
        if verbose:
            print(f"Environment {env} explained variance: {ev}")
        min_ev = min(min_ev, ev)
    if ret_ind:
        return xplvar_envs, min_ev
    return min_ev
