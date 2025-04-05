import numpy as np
import pandas as pd


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
