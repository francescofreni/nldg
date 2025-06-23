import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import geopandas as gpd
import contextily as ctx
from matplotlib.lines import Line2D


# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#         "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
#         "axes.labelsize": 12,
#         "legend.fontsize": 12,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "axes.unicode_minus": True,
#     }
# )

WIDTH, HEIGHT = 10, 6


# =====================================
# Plotting functions for the simulation
# =====================================
# def plot_max_mse(
#     max_mse_df: pd.DataFrame,
#     saveplot: bool = False,
#     nameplot: str = "max_mse",
# ) -> None:
#     color = "tab:blue"
#     n = len(max_mse_df.columns)
#     pos = np.arange(n)
#
#     means = max_mse_df.mean(axis=0)
#     stderr = max_mse_df.std(axis=0, ddof=1) / np.sqrt(len(max_mse_df))
#     lo = means - 1.96 * stderr
#     hi = means + 1.96 * stderr
#     yerr = np.vstack([means - lo, hi - means])
#
#     fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
#     vp = ax.violinplot(
#         [max_mse_df.iloc[:, i] for i in range(n)],
#         positions=pos,
#         showmeans=True,
#         showextrema=False,
#         widths=0.6,
#     )
#     for body in vp["bodies"]:
#         body.set_facecolor(color)
#         body.set_edgecolor(color)
#         body.set_alpha(0.7)
#     vp["cmeans"].set_color(color)
#     vp["cmeans"].set_linewidth(2.5)
#
#     ax.errorbar(
#         pos,
#         means,
#         yerr=yerr,
#         linestyle="none",
#         capsize=8,
#         ecolor=color,
#     )
#
#     ax.set_ylabel(r"$\mathsf{MSE}$")
#     ax.set_xticks(pos)
#     ax.set_xticklabels(
#         [
#             r"$\mathsf{RF}$",
#             r"$\mathsf{MaggingRF}$",
#             r"$\mathsf{L-MMRF}$",
#             r"$\mathsf{Post-RF}$",
#             r"$\mathsf{Post-L-MMRF}$",
#             r"$\mathsf{G-DFS-MMRF}$",
#             r"$\mathsf{G-MMRF}$",
#         ]
#     )
#     ax.grid(True, linewidth=0.2, axis="y")
#     plt.tight_layout()
#     if saveplot:
#         script_dir = os.path.dirname(__file__)
#         parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
#         results_dir = os.path.join(parent_dir, "results")
#         plots_dir = os.path.join(results_dir, "figures")
#         os.makedirs(plots_dir, exist_ok=True)
#         outpath = os.path.join(plots_dir, f"{nameplot}.png")
#         plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.show()


def plot_max_mse_boxplot(
    max_mse_df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_boxplot",
    show: bool = False,
) -> None:
    color = "tab:blue"
    n = len(max_mse_df.columns)
    pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    # Boxplot instead of violinplot
    ax.boxplot(
        [max_mse_df.iloc[:, i] for i in range(n)],
        positions=pos,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=color, color=color, alpha=0.5),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(markerfacecolor=color, marker="o", alpha=0.4),
        medianprops=dict(color="tab:blue", linewidth=2),
    )

    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [
            r"$\mathsf{RF}$",
            r"$\mathsf{MaggingRF}$",
            r"$\mathsf{L\text{-}MMRF}$",
            r"$\mathsf{Post\text{-}RF}$",
            r"$\mathsf{Post\text{-}L\text{-}MMRF}$",
            r"$\mathsf{G\text{-}DFS\text{-}MMRF}$",
            r"$\mathsf{G\text{-}MMRF}$",
        ],
    )
    ax.grid(True, linewidth=0.2, axis="y")
    plt.tight_layout()

    if saveplot:
        script_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        results_dir = os.path.join(parent_dir, "results")
        plots_dir = os.path.join(results_dir, "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_mse_envs(
    df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "mse_envs",
) -> None:
    c = ["tab:blue", "tab:orange", "tab:green"]
    env_labels = [
        r"$\mathsf{Env\ 1}$",
        r"$\mathsf{Env\ 2}$",
        r"$\mathsf{Env\ 3}$",
    ]

    methods = df["method"].unique()
    envs = sorted(df["env_id"].unique())
    n_methods = len(methods)
    n_envs = len(envs)

    fig, ax = plt.subplots(figsize=(1.5 * n_methods * n_envs, 6))

    group_spacing = n_envs
    method_tick_positions = []
    method_tick_labels = []
    violin_positions = []

    for i, method in enumerate(methods):
        for j, env_id in enumerate(envs):
            pos = i * group_spacing + j
            violin_positions.append((method, env_id, pos))

            data = df[(df["method"] == method) & (df["env_id"] == env_id)][
                "MSE"
            ]
            vp = ax.violinplot(
                data,
                positions=[pos],
                showmeans=True,
                showextrema=False,
                widths=0.7,
            )

            color = c[j]
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.7)

            vp["cmeans"].set_color(c[j])
            vp["cmeans"].set_linewidth(2)

        mid = i * group_spacing + (n_envs - 1) / 2
        method_tick_positions.append(mid)
        method_tick_labels.append(rf"$\mathsf{{{method}}}$")

    # Force rendering to get updated axis limits
    fig.canvas.draw()
    ymin, ymax = ax.get_ylim()

    # Add environment and method labels
    for method, env_id, pos in violin_positions:
        ax.text(
            pos,
            ymin - 0.02 * (ymax - ymin),
            env_labels[env_id],
            ha="center",
            va="top",
            fontsize=10,
        )

    for i, mid in enumerate(method_tick_positions):
        ax.text(
            mid,
            ymin - 0.08 * (ymax - ymin),
            method_tick_labels[i],
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    # Hide default xticks
    ax.set_xticks([])

    # Separator lines between method groups
    for i in range(1, n_methods):
        ax.axvline(
            i * group_spacing - 0.5,
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.4,
        )

    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.grid(True, linewidth=0.2, axis="y")

    plt.subplots_adjust(top=0.88, bottom=0.25)
    plt.tight_layout()
    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        results_dir = os.path.join(parent_dir, "results")
        plots_dir = os.path.join(results_dir, "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_max_mse_msl(
    res: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_msl",
    show: bool = False,
) -> None:
    colors = ["lightskyblue", "orange", "mediumpurple"]
    methods = ["RF", "Post-RF", "MaggingRF"]
    min_samples_leaf = np.unique(res["min_samples_leaf"])
    nsim = res.shape[0] / len(methods)

    all_stats = []
    for method in methods:
        values = res[res["Method"] == method]
        for i, msl in enumerate(min_samples_leaf):
            maxmse_values = values[values["min_samples_leaf"] == msl]["MSE"]
            mean_val = np.mean(maxmse_values)
            std_err = np.std(maxmse_values, ddof=1) / np.sqrt(nsim)
            width = 1.96 * std_err
            all_stats.append(
                {
                    "msl": msl,
                    "method": method,
                    "mean": mean_val,
                    "lower": mean_val - width,
                    "upper": mean_val + width,
                }
            )

    df = pd.DataFrame(all_stats)

    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        plt.plot(
            method_df["msl"],
            method_df["mean"],
            label=method,
            color=colors[i],
            marker="o",
            markeredgecolor="white",
        )
        plt.fill_between(
            method_df["msl"],
            method_df["lower"],
            method_df["upper"],
            alpha=0.3,
            color=colors[i],
        )

    plt.xlabel("Minimum number of observations per leaf")
    plt.ylabel("Maximum MSE over training environments")
    plt.legend()
    plt.grid(True, linewidth=0.2)
    plt.tight_layout()
    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        results_dir = os.path.join(parent_dir, "results")
        plots_dir = os.path.join(results_dir, "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


# ============================================
# Plotting functions for the real data example
# ============================================
def plot_max_score_housing(
    df: pd.DataFrame,
    metric: str = "Test_MSE",
    saveplot: bool = False,
    nameplot: str = "mse_heldout",
    setting: int = 1,
    show: bool = False,
) -> None:
    QUADRANTS = (
        ["SW", "SE", "NW", "NE"]
        if setting == 1
        else ["Env 1", "Env 2", "Env 3", "Env 4"]
    )

    # Dynamically get model list
    num_models = len(df["Model"].unique())
    models = (
        ["RF", "Post-RF"]
        if num_models == 2
        else ["RF", "Post-RF-MSE", "Post-RF-Regret"]
    )
    num_models = len(models)
    if num_models not in [2, 3]:
        raise ValueError("Expected 2 or 3 models in the data.")

    # Colors and offsets
    default_colors = ["lightskyblue", "orange", "plum"]
    colors = default_colors[:num_models]
    delta = 0.1
    offsets = np.linspace(-delta, +delta, num_models)

    # Compute group stats
    grp = df.groupby(["HeldOutQuadrant", "Model"])[metric]
    means = grp.mean().unstack().reindex(QUADRANTS)
    stds = grp.std().unstack().reindex(QUADRANTS)
    counts = grp.count().unstack().reindex(QUADRANTS)
    ci95 = 1.96 * stds / np.sqrt(counts)

    x0 = np.arange(len(QUADRANTS))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each model
    for idx, (off, model) in enumerate(zip(offsets, models)):
        xm = x0 + off
        ax.errorbar(
            xm,
            means[model],
            yerr=ci95[model],
            fmt="o",
            color=colors[idx],
            markersize=8,
            markeredgewidth=0,
            elinewidth=2.5,
            capsize=0,
            label=model,
        )

    ax.set_xticks(x0)
    ax.set_xticklabels(QUADRANTS)
    ax.set_xlabel("Held-Out Quadrant")
    ax.set_ylabel(r"$\mathsf{MSPE}$")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)

    plt.tight_layout()
    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def plot_score_envs_housing(
    df_env_spec: pd.DataFrame,
    df_main: pd.DataFrame | None = None,
    saveplot: bool = False,
    nameplot: str = "env_specific_mse",
    setting: int = 1,
    score: str = "mse",
    show: bool = False,
):
    QUADRANTS = (
        ["SW", "SE", "NW", "NE"]
        if setting == 1
        else ["Env 1", "Env 2", "Env 3", "Env 4"]
    )
    target_score = "MSE" if score == "mse" else "Regret"

    # Dynamically get model list
    num_models = len(df_env_spec["Model"].unique())
    models = (
        ["RF", "Post-RF"]
        if num_models == 2
        else ["RF", "Post-RF-MSE", "Post-RF-Regret"]
    )
    num_models = len(models)
    if num_models not in [2, 3]:
        raise ValueError("Expected 2 or 3 models.")

    # Colors and offsets
    default_colors = ["lightskyblue", "orange", "plum"]
    colors = default_colors[:num_models]
    delta = 0.12
    n_subenv = 3
    figsize = (12, 6)

    fig, ax = plt.subplots(figsize=figsize)
    seen = {m: False for m in models}
    label_positions = []

    for i, ho in enumerate(QUADRANTS):
        subenvs = [q for q in QUADRANTS if q != ho]

        for j, env in enumerate(subenvs):
            x_base = i * n_subenv + j
            label_positions.append((ho, env, x_base))

            for m_idx, model in enumerate(models):
                ser = df_env_spec[
                    (df_env_spec["HeldOutQuadrant"] == ho)
                    & (df_env_spec["Model"] == model)
                    & (df_env_spec["EnvIndex"] == QUADRANTS.index(env))
                ][target_score]

                if ser.empty:
                    continue

                mean = ser.mean()
                std = ser.std(ddof=1)
                ci95 = 1.96 * std / np.sqrt(ser.count())

                x = x_base + (m_idx - (num_models - 1) / 2) * delta
                label = model if not seen[model] else "_nolegend_"
                seen[model] = True

                ax.errorbar(
                    x,
                    mean,
                    yerr=ci95,
                    fmt="o",
                    color=colors[m_idx],
                    markersize=8,
                    elinewidth=2.5,
                    capsize=0,
                    label=label,
                )

    # Vertical separators between held-out quadrants
    for k in range(1, len(QUADRANTS)):
        sep_x = k * n_subenv - 0.5
        ax.axvline(sep_x, linewidth=0.5, color="black")

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(QUADRANTS) * n_subenv - 0.5)

    fig.canvas.draw()
    y0, y1 = ax.get_ylim()

    # Sub-env labels
    for _, env, x in label_positions:
        ax.text(
            x,
            y0 - 0.02 * (y1 - y0),
            rf"$\mathsf{{{env}}}$",
            ha="center",
            va="top",
            fontsize=10,
        )

    # Held-out quadrant labels
    for i, ho in enumerate(QUADRANTS):
        mid = i * n_subenv + (n_subenv - 1) / 2
        ax.text(
            mid,
            y0 - 0.08 * (y1 - y0),
            rf"$\mathsf{{{ho}}}$",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    # Add horizontal lines from df_main
    if score == "mse":
        for i, ho in enumerate(QUADRANTS):
            for m_idx, model in enumerate(models):
                df_sub = df_main[
                    (df_main["HeldOutQuadrant"] == ho)
                    & (df_main["Model"] == model)
                ]

                if df_sub.empty:
                    continue

                train_mean = df_sub["Train_MSE"].mean()
                train_std = df_sub["Train_MSE"].std(ddof=1)
                ci95 = 1.96 * train_std / np.sqrt(len(df_sub))

                # Compute horizontal span range for this quadrant
                start = i * n_subenv - 0.4
                end = (i + 1) * n_subenv - 0.6

                label = (
                    f"{model} Overall MSE"
                    if (i == len(QUADRANTS) - 1)
                    else "_nolegend_"
                )
                ax.hlines(
                    y=train_mean,
                    xmin=start,
                    xmax=end,
                    color=colors[m_idx],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=label,
                )

                x_min, x_max = ax.get_xlim()
                xmin_norm = (start - x_min) / (x_max - x_min)
                xmax_norm = (end - x_min) / (x_max - x_min)

                ax.axhspan(
                    train_mean - ci95,
                    train_mean + ci95,
                    xmin=xmin_norm,
                    xmax=xmax_norm,
                    color=colors[m_idx],
                    alpha=0.15,
                )

    ax.legend(loc="best", frameon=True)
    lab = r"$\mathsf{MSE}$" if score == "mse" else r"$\mathsf{Regret}$"
    ax.set_ylabel(lab)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)

    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.tight_layout()

    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plots_dir, f"{nameplot}.png"),
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()


# def plot_mse_envs_housing_bootstrap(
#     df: pd.DataFrame,
#     ci_type: str = "perc",
#     saveplot: bool = False,
#     nameplot: str = "bootstrap_metrics",
# ) -> None:
#     QUADRANTS = ["SW", "SE", "NW", "NE"]
#     MODELS = ["RF", "Post-RF"]
#     COLORS = {"RF": "lightskyblue", "Post-RF": "orange"}
#     OFFSETS = {"RF": -0.1, "Post-RF": 0.1}
#
#     x_pos = np.arange(len(QUADRANTS))
#     fig, ax = plt.subplots(figsize=(8, 5))
#
#     for model in MODELS:
#         means, err_low, err_high = [], [], []
#         for quadrant in QUADRANTS:
#             row = df[
#                 (df["Method"] == model) & (df["Quadrant"] == quadrant)
#             ].iloc[0]
#             mean = row["MSE_mean"]
#             lo = row[f"Lower_CI_{ci_type}"]
#             hi = row[f"Upper_CI_{ci_type}"]
#             means.append(mean)
#             err_low.append(max(mean - lo, 0))
#             err_high.append(max(hi - mean, 0))
#
#         x_model = x_pos + OFFSETS[model]
#         yerr = [err_low, err_high]
#         ax.errorbar(
#             x_model,
#             means,
#             yerr=yerr,
#             fmt="o",
#             color=COLORS[model],
#             markersize=10,
#             markeredgewidth=0,
#             elinewidth=2.5,
#             capsize=0,
#             label=model,
#         )
#
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(QUADRANTS)
#     ax.set_xlabel("Quadrant")
#     ax.set_ylabel("MSE")
#     ax.legend(loc="upper right")
#     ax.grid(axis="y", linestyle="--", alpha=0.5)
#
#     plt.tight_layout()
#     if saveplot:
#         plots_dir = os.path.join("results", "figures")
#         os.makedirs(plots_dir, exist_ok=True)
#         outpath = os.path.join(plots_dir, f"{nameplot}.png")
#         plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.show()


def plot_mse_envs_housing_resample(
    df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "mse_envs_resample",
) -> None:
    QUADRANTS = ["SW", "SE", "NW", "NE"]
    df_long = df.melt(
        id_vars="method",
        value_vars=QUADRANTS,
        var_name="Environment",
        value_name="maxMSE",
    )
    grp = df_long.groupby(["Environment", "method"])["maxMSE"]
    means = grp.mean().unstack().reindex(QUADRANTS)
    stds = grp.std().unstack().reindex(QUADRANTS)
    counts = grp.count().unstack().reindex(QUADRANTS)
    ci95 = 1.96 * stds / np.sqrt(counts)

    # Calculate overall means and confidence intervals for each method
    overall_grp = df.groupby("method")["Overall"]
    overall_means = overall_grp.mean()
    overall_stds = overall_grp.std()
    overall_counts = overall_grp.count()
    overall_ci95 = 1.96 * overall_stds / np.sqrt(overall_counts)

    x0 = np.arange(len(QUADRANTS))
    models = ["RF", "Post-RF"]
    colors = ["lightskyblue", "orange"][: len(models)]
    delta = 0.1
    offsets = np.linspace(-delta, +delta, len(models))
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (model, off) in enumerate(zip(models, offsets)):
        ax.errorbar(
            x0 + off,
            means[model],
            yerr=ci95[model],
            fmt="o",
            color=colors[idx],
            markersize=8,
            markeredgewidth=0,
            elinewidth=2.5,
            capsize=0,
            label=model,
        )

        # Add horizontal line and confidence band for overall mean
        if model in overall_means.index:
            overall_mean = overall_means[model]
            overall_ci = overall_ci95[model]

            # Horizontal line
            ax.axhline(
                y=overall_mean,
                color=colors[idx],
                linestyle="--",
                alpha=0.8,
                linewidth=2,
                label=f"{model} Overall MSE",
            )

            # Confidence band
            ax.axhspan(
                ymin=overall_mean - overall_ci,
                ymax=overall_mean + overall_ci,
                color=colors[idx],
                alpha=0.15,
            )

    ax.set_xticks(x0)
    ax.set_xticklabels(QUADRANTS)
    ax.set_xlabel("Environment")
    ax.set_ylabel("MSE")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)
    plt.tight_layout()
    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_max_mse_mtry(
    res: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_mtry",
) -> None:
    cols = ["maxMSE_RF", "maxMSE_Post-RF"]
    labels = ["RF", "Post-RF"]
    colors = ["lightskyblue", "orange"]

    plt.figure(figsize=(8, 5))
    for col, label, c in zip(cols, labels, colors):
        plt.plot(
            res["mtry"],
            res[col],
            marker="o",
            linestyle="-",
            label=label,
            color=c,
            markeredgecolor="white",
        )

    plt.xlabel(r"$m_{\mathrm{try}}$")
    plt.ylabel("Maximum MSE over training environments")
    plt.grid(True, linewidth=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()

    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


def plot_max_mse_mtry_resample(
    res: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_mtry_resample",
    show: bool = False,
) -> None:
    methods = ["RF", "Post-RF"]
    colors = ["lightskyblue", "orange"]

    nsim = res.groupby(["method", "mtry"]).size().iloc[0]

    stats = []
    for method in methods:
        df_m = res[res["method"] == method]
        for m in sorted(df_m["mtry"].unique()):
            vals = df_m[df_m["mtry"] == m]["maxMSE"]
            mean = vals.mean()
            stderr = vals.std(ddof=1) / np.sqrt(nsim)
            width = 1.96 * stderr
            stats.append(
                {
                    "method": method,
                    "mtry": m,
                    "mean": mean,
                    "lower": mean - width,
                    "upper": mean + width,
                }
            )

    df_stats = pd.DataFrame(stats)

    plt.figure(figsize=(8, 5))
    for method, color in zip(methods, colors):
        df_m = df_stats[df_stats["method"] == method]
        plt.plot(
            df_m["mtry"],
            df_m["mean"],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            markeredgecolor="white",
        )
        plt.fill_between(
            df_m["mtry"],
            df_m["lower"],
            df_m["upper"],
            color=color,
            alpha=0.3,
        )

    plt.xlabel(r"$m_{\mathrm{try}}$")
    plt.ylabel("Maximum MSE over training environments")
    plt.grid(True, linewidth=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()

    if saveplot:
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        plots_dir = os.path.join(parent_dir, "results", "figures")
        os.makedirs(plots_dir, exist_ok=True)
        outpath = os.path.join(plots_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_quadrants_with_basemap():
    def assign_quadrant(Z: pd.DataFrame) -> pd.Series:
        lat = Z["Latitude"]
        lon = Z["Longitude"]

        west = lon < -121
        east = ~west  # same as lon >= -121

        # For west side: split at 38
        sw = (lat < 38) & west
        nw = (lat >= 38) & west

        # For east side: split at 34.5 or 36
        se = (lat < 34.5) & east
        ne = (lat >= 34.5) & east

        quadrant = pd.Series(index=Z.index, dtype="Int64")
        quadrant[sw] = 0  # SW
        quadrant[se] = 1  # SE
        quadrant[nw] = 2  # NW
        quadrant[ne] = 3  # NE

        if quadrant.isna().any():
            raise ValueError("Some rows failed to get a quadrant label.")

        return quadrant.rename("env_quadrant")

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    Z = X[["Latitude", "Longitude"]]
    X = X.drop(["Latitude", "Longitude"], axis=1)
    env = assign_quadrant(Z)

    df = Z.copy()
    df["quadrant"] = env
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    colors = {0: "lightskyblue", 1: "orange", 2: "plum", 3: "yellowgreen"}
    labels = {
        0: rf"Env 1: $\bar{{y}}$ = {round(np.mean(y[env==0]), 2)}",
        1: rf"Env 2: $\bar{{y}}$ = {round(np.mean(y[env==1]), 2)}",
        2: rf"Env 3: $\bar{{y}}$ = {round(np.mean(y[env==2]), 2)}",
        3: rf"Env 4: $\bar{{y}}$ = {round(np.mean(y[env==3]), 2)}",
    }

    fig, ax = plt.subplots(figsize=(8, 10))
    for q in range(4):
        gdf[gdf["quadrant"] == q].plot(
            ax=ax, markersize=3, color=colors[q], label=labels[q], alpha=0.3
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=labels[q],
            markerfacecolor=colors[q],
            markersize=10,
            alpha=0.6,
        )
        for q in range(4)
    ]
    ax.legend(handles=legend_elements, title="Quadrant", loc="best")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    plt.tight_layout()
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    plots_dir = os.path.join(parent_dir, "results", "figures")
    os.makedirs(plots_dir, exist_ok=True)
    outpath = os.path.join(plots_dir, "quadrants_plot")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
