import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

NAME_RF = "MaxRM-RF"
NAME_SS = "MaxRM-SS"

COLORS = {
    "blue": "#5790FC",
    "orange": "#F89C20",
    "purple": "#964A8B",
    "red": "#E42536",
    "lightblue": "#86C8DD",
    "purple2": "#7A21DD",
    "tan": "#B9AC70",
    "brown": "#A96B59",
}


# ===============
# DATA GENERATION
# ===============
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
    n: int = 500,
    random_state: int = 0,
    setting: int = 1,
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
    n_easy: int = 300,
    n_hard: int = 300,
    random_state: int = 0,
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
    noise_var_env2: float = 10.0,
    random_state: int = 0,
    setting: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    if setting == 1:
        X = rng.uniform(-5, 5, size=(n_samples, 1))
        y = 2.0 * X.squeeze() + 1.0 + rng.normal(0, 1.0, size=n_samples)
        n_adv = int(adv_fraction * n_samples)
        adv_indices = rng.choice(n_samples, n_adv, replace=False)
        y[adv_indices] += rng.normal(0, noise_var_env2, size=n_adv) + 0.5 * (
            X[adv_indices].squeeze() ** 2
        )  # + 10
    else:
        X = rng.uniform(0, 2, size=(n_samples, 1))
        y = X.squeeze() + rng.normal(0, 0.5, size=n_samples)
        n_adv = int(adv_fraction * n_samples)
        adv_indices = rng.choice(n_samples, n_adv, replace=False)

        y[adv_indices] += (
            rng.normal(0, noise_var_env2, size=n_adv)
            + 0.5 * X[adv_indices].squeeze() ** 2
            + X[adv_indices].squeeze()
        ) + 2

    env = np.zeros(n_samples, dtype=int)
    env[adv_indices] = 1
    df = pd.DataFrame({"X": X.squeeze(), "Y": y, "E": env})

    return df


def gen_data_v6(
    n: int = 300,
    random_state: int = 0,
    noise_std: float = 0.2,
    new_x: bool = False,
    setting: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_e = n // 3

    X = rng.uniform(-4, 4, size=n_e)
    if setting == 1:
        Y = np.where(X <= 0, X / 2, 3 * X) + rng.normal(0, noise_std, size=n_e)
    else:
        Y = np.where(X <= 0, -X / 2, 4 * X) + rng.normal(
            0, noise_std, size=n_e
        )
    df1 = pd.DataFrame({"X": X, "Y": Y, "E": 0})

    if new_x:
        X = rng.uniform(-4, 4, size=n_e)
    Y = np.where(X <= 0, 3 * X, X / 2) + rng.normal(0, noise_std, size=n_e)
    df2 = pd.DataFrame({"X": X, "Y": Y, "E": 1})

    if new_x:
        X = rng.uniform(-4, 4, size=n_e)
    Y = np.where(X <= 0, 2.5 * X, X) + rng.normal(0, noise_std, size=n_e)
    df3 = pd.DataFrame({"X": X, "Y": Y, "E": 2})

    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    return df_all


def gen_data_v7(
    n: int = 300,
    random_state: int = 0,
    new_x: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_e = n // 3
    noise_std = 0.1

    X = rng.uniform(-4, 4, size=n_e)
    Y = np.where(X <= 0, -(X + 2) / 2, (X - 2) / 2) + rng.normal(
        0, noise_std, size=n_e
    )
    df1 = pd.DataFrame({"X": X, "Y": Y, "E": 0})

    if new_x:
        X = rng.uniform(-4, 4, size=n_e)
    Y = np.where(X <= 0, X + 2, -X + 2) + rng.normal(0, noise_std, size=n_e)
    df2 = pd.DataFrame({"X": X, "Y": Y, "E": 1})

    if new_x:
        X = rng.uniform(-4, 4, size=n_e)
    Y = np.zeros(len(X))
    Y[X <= -2] = -(X[X <= -2] + 4) / 2
    Y[(X > -2) & (X <= 2)] = X[(X > -2) & (X <= 2)] / 2
    Y[X > 2] = -(X[X > 2] - 4) / 2
    Y += rng.normal(0, noise_std, size=n_e)
    df3 = pd.DataFrame({"X": X, "Y": Y, "E": 2})

    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    return df_all


def gen_data_v8(
    n: int = 300,
    random_state: int = 0,
    new_x: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_e = n // 3
    noise_std = 0.1

    X = rng.uniform(-np.pi, np.pi, size=n_e)
    Y = np.sin(X) + rng.normal(0, noise_std, size=n_e)
    df1 = pd.DataFrame({"X": X, "Y": Y, "E": 0})

    if new_x:
        X = rng.uniform(-np.pi, np.pi, size=n_e)
    Y = 2 * np.sin(X + 2 * np.pi / 3) + rng.normal(0, noise_std, size=n_e)
    df2 = pd.DataFrame({"X": X, "Y": Y, "E": 1})

    if new_x:
        X = rng.uniform(-np.pi, np.pi, size=n_e)
    Y = np.sin(X + 4 * np.pi / 3) + rng.normal(0, noise_std, size=n_e)
    df3 = pd.DataFrame({"X": X, "Y": Y, "E": 2})

    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    return df_all


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


def min_reward(
    Ytrue: np.ndarray,
    Ypred: np.ndarray,
    Env: np.ndarray,
    verbose: bool = False,
    ret_ind: bool = False,
) -> float | tuple[list, float]:
    """
    Compute the minimum reward across environments.

    For each environment, the reward is computed as:
        EV = mean(Ytrue^2) - mean((Ytrue - Ypred)^2)
    This function returns the minimum reward across all environments.

    Args:
        Ytrue (np.ndarray): True target values.
        Ypred (np.ndarray): Predicted target values.
        Env (np.ndarray): Environment labels for each sample.
        verbose (bool): Whether to print the reward for each environment.
        ret_ind (bool): Whether to return also the reward for each environment.
        demean (bool): Whether the response was centered in each environment.

    Returns:
        if ret_ind:
            reward_envs (list): Reward for each environment.
        min_reward (float): Minimum reward across environments.
    """
    min_reward = float("inf")
    reward_envs = []
    for env in np.unique(Env):
        Ytrue_e = Ytrue[Env == env]
        Ypred_e = Ypred[Env == env]
        reward = np.mean(Ytrue_e**2) - np.mean((Ytrue_e - Ypred_e) ** 2)
        # reward = np.var(Ytrue_e) - np.var(Ytrue_e - Ypred_e)
        reward_envs.append(reward)
        if verbose:
            print(f"Environment {env} reward: {reward}")
        min_reward = min(min_reward, reward)
    if ret_ind:
        return reward_envs, min_reward
    return min_reward


def max_regret(
    Ytrue: np.ndarray,
    Ypred: np.ndarray,
    Yerm: np.ndarray,
    Env: np.ndarray,
    verbose: bool = False,
    ret_ind: bool = False,
) -> float | tuple[list, float]:
    """
    Compute the maximum regret across environments.

    Args:
        Ytrue (array): True target values.
        Ypred (array): Predicted target values.
        Yerm (array): Predicted target values with ERM.
        Env (array): Environment values.
        verbose (bool): Whether to print the MSE for each environment.
        ret_ind (bool): Whether to return also the MSE for each environment.

    Returns:
        if ret_ind:
            mse_envs (list): MSE for each environment.
        maxmse (float): Maximum mean squared error.
    """
    mregret = 0.0
    regret_envs = []
    for env in np.unique(Env):
        Ytrue_e = Ytrue[Env == env]
        Ypred_e = Ypred[Env == env]
        Yerm_e = Yerm[Env == env]
        regret = np.mean((Ytrue_e - Ypred_e) ** 2 - (Ytrue_e - Yerm_e) ** 2)
        regret_envs.append(regret)
        if verbose:
            print(f"Environment {env} regret: {regret}")
        mregret = max(mregret, regret)
    if ret_ind:
        return regret_envs, mregret
    return mregret


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


# ========
# PLOTTING
# ========
def plot_dtr(
    dtr: pd.DataFrame,
    optfun: int | None = None,
    gdro: bool = False,
    obj_comparison: bool = False,
    saveplot: bool = False,
    nameplot: str = "setting5",
    suffix: str | None = None,
    legend_pos: None | str = "upper left",
):
    """
    Plotting function for random forest results
    """
    data_colors = ["black", "grey", "silver"]
    environments = sorted(dtr["E"].unique())

    if "y_clean" in dtr:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    for idx, env in enumerate(environments):
        marker_style = "o"
        ax.scatter(
            dtr[dtr["E"] == env]["X"],
            dtr[dtr["E"] == env]["Y"],
            color=data_colors[idx],
            marker=marker_style,
            alpha=0.5,
            s=30,
            label=f"Env {env + 1}",
        )
    if not obj_comparison:
        if not gdro:
            ax.plot(
                dtr["X_sorted"],
                dtr["fitted_rf"],
                color=COLORS["blue"],
                linewidth=2,
                label="RF",
            )
            if suffix is None:
                lab = (
                    f"{NAME_RF}(posthoc-mse)"
                    if "fitted_minmax_xtrgrd" in dtr
                    else f"{NAME_RF}(mse)"
                )
            else:
                lab = f"{NAME_RF}({suffix})"
            ax.plot(
                dtr["X_sorted"],
                dtr["fitted_minmax"],
                color=COLORS["orange"],
                linewidth=2,
                label=lab,
            )
            if "y_clean" in dtr:
                y_clean = np.array(dtr["y_clean"]).ravel()
                X = np.array(dtr["X"][dtr["E"] == 0]).ravel()
                env_label = np.array(dtr["E"]).ravel()
                sorted_idx = np.argsort(X)
                X_sorted = X[sorted_idx]
                ax.plot(
                    X_sorted,
                    y_clean[env_label == 0][sorted_idx],
                    color=COLORS["purple"],
                    linewidth=2,
                    label="$f^{e_1}$",
                    linestyle="--",
                )
                ax.plot(
                    X_sorted,
                    y_clean[env_label == 1][sorted_idx],
                    color=COLORS["red"],
                    linewidth=2,
                    label="$f^{e_2}$",
                    linestyle="--",
                )
                ax.plot(
                    X_sorted,
                    y_clean[env_label == 2][sorted_idx],
                    color=COLORS["lightblue"],
                    linewidth=2,
                    label="$f^{e_3}$",
                    linestyle="--",
                )
            if "fitted_minmax_xtrgrd" in dtr:
                ax.plot(
                    dtr["X_sorted"],
                    dtr["fitted_minmax_xtrgrd"],
                    color=COLORS["purple"],
                    linewidth=2,
                    label=f"{NAME_RF}(posthoc-mse-xtrgrd)",
                )
            if "fitted_magging" in dtr:
                ax.plot(
                    dtr["X_sorted"],
                    dtr["fitted_magging"],
                    color=COLORS["purple"],
                    linewidth=2,
                    label="RF(magging)",
                )
        else:
            ax.plot(
                dtr["X_sorted"],
                dtr["fitted_default"],
                color=COLORS["blue"],
                linewidth=2,
                label="NN",
            )
            ax.plot(
                dtr["X_sorted"],
                dtr["fitted_gdro"],
                color=COLORS["orange"],
                linewidth=2,
                label="NN-GDRO",
            )
    else:
        ax.plot(
            dtr["X_sorted"],
            dtr["fitted_rf"],
            color=COLORS["blue"],
            linewidth=2,
            label="RF",
        )
        ax.plot(
            dtr["X_sorted"],
            dtr["fitted_mse"],
            color=COLORS["orange"],
            linewidth=2,
            label=f"{NAME_RF}(posthoc-mse)",
        )
        ax.plot(
            dtr["X_sorted"],
            dtr["fitted_nrw"],
            color=COLORS["purple"],
            linewidth=2,
            label=f"{NAME_RF}(posthoc-nrw)",
        )
        ax.plot(
            dtr["X_sorted"],
            dtr["fitted_regret"],
            color=COLORS["red"],
            linewidth=2,
            label=f"{NAME_RF}(posthoc-reg)",
        )
        if "fitted_magging" in dtr:
            ax.plot(
                dtr["X_sorted"],
                dtr["fitted_magging"],
                color=COLORS["lightblue"],
                linewidth=2,
                label="RF(magging)",
            )
        if "y_clean" in dtr:
            y_clean = np.array(dtr["y_clean"]).ravel()
            X = np.array(dtr["X"][dtr["E"] == 0]).ravel()
            env_label = np.array(dtr["E"]).ravel()
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            ax.plot(
                X_sorted,
                y_clean[env_label == 0][sorted_idx],
                color=COLORS["lightblue"],
                linewidth=2,
                label="$f^{e_1}$",
                linestyle="--",
            )
            ax.plot(
                X_sorted,
                y_clean[env_label == 1][sorted_idx],
                color=COLORS["purple2"],
                linewidth=2,
                label="$f^{e_2}$",
                linestyle="--",
            )
            ax.plot(
                X_sorted,
                y_clean[env_label == 2][sorted_idx],
                color=COLORS["tan"],
                linewidth=2,
                label="$f^{e_3}$",
                linestyle="--",
            )

    if optfun is not None:
        x_range = np.linspace(
            dtr["X_sorted"].min(), dtr["X_sorted"].max(), 1000
        )

        if optfun == 1:
            y_opt = 0.8 * np.sin(x_range / 2) ** 2 + 3
        elif optfun == 2:
            y_opt = np.where(x_range > 0, 2.4 * x_range, -2.4 * x_range)
        elif optfun == 3:
            y_opt = np.where(x_range > 0, 1.75 * x_range, 1.75 * x_range)
        elif optfun == 4:
            y_opt = np.where(x_range > 0, 2.25 * x_range, 1.25 * x_range)
        else:
            y_opt = None

        if y_opt is not None:
            ax.plot(
                x_range,
                y_opt,
                color=COLORS["red"],
                linewidth=3,
                label="Oracle",
                linestyle="--",
            )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc=legend_pos)

    plt.tight_layout()
    if saveplot:
        script_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_dtr_ss(
    dtr: pd.DataFrame,
    x_grid: np.ndarray,
    preds_erm: np.ndarray,
    preds_mse: np.ndarray,
    preds_nrw: np.ndarray | None = None,
    preds_regret: np.ndarray | None = None,
    preds_magging: np.ndarray | None = None,
    obj_comparison: bool = False,
    optfun: int | None = None,
    saveplot: bool = False,
    nameplot: str = "setting5",
):
    """
    Plotting function for smoothing splines results
    """
    data_colors = ["black", "grey", "silver"]
    environments = sorted(dtr["E"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, env in enumerate(environments):
        marker_style = "o"
        ax.scatter(
            dtr[dtr["E"] == env]["X"],
            dtr[dtr["E"] == env]["Y"],
            color=data_colors[idx],
            marker=marker_style,
            alpha=0.5,
            s=30,
            label=f"Env {env + 1}",
        )

    ax.plot(x_grid, preds_erm, color=COLORS["blue"], linewidth=2, label="SS")
    ax.plot(
        x_grid,
        preds_mse,
        color=COLORS["orange"],
        linewidth=2,
        label=f"{NAME_SS}(mse)",
    )
    if preds_nrw is not None:
        ax.plot(
            x_grid,
            preds_nrw,
            color=COLORS["purple"],
            linewidth=2,
            label=f"{NAME_SS}(nrw)",
        )
    if preds_regret is not None:
        ax.plot(
            x_grid,
            preds_regret,
            color=COLORS["red"],
            linewidth=2,
            label=f"{NAME_SS}(reg)",
        )
    if preds_magging is not None:
        col_idx = "lightblue" if obj_comparison else "purple"
        ax.plot(
            x_grid,
            preds_magging,
            color=COLORS[col_idx],
            linewidth=2,
            label="SS(magging)",
        )

    if optfun is not None:
        if optfun == 1:
            y_opt = 0.8 * np.sin(x_grid / 2) ** 2 + 3
        elif optfun == 2:
            y_opt = np.where(x_grid > 0, 2.4 * x_grid, -2.4 * x_grid)
        elif optfun == 3:
            y_opt = np.where(x_grid > 0, 1.75 * x_grid, 1.75 * x_grid)
        elif optfun == 4:
            y_opt = np.where(x_grid > 0, 2.25 * x_grid, 1.25 * x_grid)
        else:
            y_opt = None

        if y_opt is not None:
            ax.plot(
                x_grid,
                y_opt,
                color=COLORS["red"],
                linewidth=3,
                label="Oracle",
                linestyle="--",
            )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    if saveplot:
        script_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_dtr_all_methods(
    dtr: pd.DataFrame,
    optfun: int | None = None,
    saveplot: bool = False,
    nameplot: str = "setting5_allmethods",
):
    line_colors = [
        "#5790FC",
        "#F89C20",
        "#964A8B",
        "#86C8DD",
        "#7A21DD",
        "#B9AC70",
        "#A96B59",
    ]
    data_colors = ["black", "grey", "silver"]
    environments = sorted(dtr["E"].unique())

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, env in enumerate(environments):
        marker_style = "o"
        ax.scatter(
            dtr[dtr["E"] == env]["X"],
            dtr[dtr["E"] == env]["Y"],
            color=data_colors[idx],
            marker=marker_style,
            alpha=0.5,
            s=30,
            label=f"Env {env + 1}",
        )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_rf"],
        color=line_colors[0],
        linewidth=2,
        label="RF",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_magging"],
        color=line_colors[1],
        linewidth=2,
        label="RF(magging)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_l_mmrf"],
        color=line_colors[2],
        linewidth=2,
        label=f"{NAME_RF}(local)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_post_rf"],
        color=line_colors[3],
        linewidth=2,
        label=f"{NAME_RF}(posthoc)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_post_l_mmrf"],
        color=line_colors[4],
        linewidth=2,
        label=f"{NAME_RF}(posthoc-local)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_g_dfs_mmrf"],
        color=line_colors[5],
        linewidth=2,
        label=f"{NAME_RF}(global-dfs)",
    )
    ax.plot(
        dtr["X_sorted"],
        dtr["fitted_g_mmrf"],
        color=line_colors[6],
        linewidth=2,
        label=f"{NAME_RF}(global)",
    )

    if optfun is not None:
        x_range = np.linspace(
            dtr["X_sorted"].min(), dtr["X_sorted"].max(), 1000
        )

        if optfun == 1:
            y_opt = 0.8 * np.sin(x_range / 2) ** 2 + 3
        elif optfun == 2:
            y_opt = np.where(x_range > 0, 2.4 * x_range, -2.4 * x_range)
        elif optfun == 3:
            y_opt = np.where(x_range > 0, 1.75 * x_range, 1.75 * x_range)
        elif optfun == 4:
            y_opt = np.where(x_range > 0, 2.25 * x_range, 1.25 * x_range)
        else:
            y_opt = None

        if y_opt is not None:
            ax.plot(
                x_range,
                y_opt,
                color="#E42536",
                linewidth=3,
                label="Oracle",
                linestyle="--",
            )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()

    if saveplot:
        script_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_tricontour(diff_map, metric):
    q1_grid, q2_grid, diff_grid = [], [], []
    for (q1, q2), diffs in diff_map.items():
        q1_grid.append(q1)
        q2_grid.append(q2)
        diff_grid.append(np.mean(diffs))

    q1_grid = np.array(q1_grid)
    q2_grid = np.array(q2_grid)
    diff_grid = np.array(diff_grid)

    # Convert to barycentric (equilateral triangle) coordinates
    q3_grid = 1 - q1_grid - q2_grid
    x = q2_grid + q3_grid / 2.0
    y = (np.sqrt(3) / 2.0) * q3_grid

    # Plot equilateral heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.tricontourf(x, y, diff_grid, levels=30, cmap="Blues")

    # Triangle boundary
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="black", lw=1.5)

    # Add labels for vertices
    ax.text(
        0.0, -0.02, r"$q_1=1$", ha="center", va="top", fontsize=12
    )  # Bottom-left vertex
    ax.text(
        1.0, -0.02, r"$q_2=1$", ha="center", va="top", fontsize=12
    )  # Bottom-right vertex
    ax.text(
        0.5,
        np.sqrt(3) / 2 + 0.02,
        r"$q_3=1$",
        ha="center",
        va="bottom",
        fontsize=12,
    )  # Top vertex

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.05)
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

    ax.scatter(
        [0, 1, 0.5],
        [0, 0, np.sqrt(3) / 2],
        color="black",
        s=40,
        linewidths=1.0,
        zorder=100,
        clip_on=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
