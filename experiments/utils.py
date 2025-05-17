import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.unicode_minus": True,
    }
)

WIDTH, HEIGHT = 10, 6


def plot_max_mse_with_ci(
    max_mse_df,
    saveplot=False,
    nameplot="max_mse",
) -> None:
    color = "tab:blue"
    n = len(max_mse_df.columns)
    pos = np.arange(n)

    means = max_mse_df.mean(axis=0)
    stderr = max_mse_df.std(axis=0, ddof=1) / np.sqrt(len(max_mse_df))
    lo = means - 1.96 * stderr
    hi = means + 1.96 * stderr
    yerr = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    vp = ax.violinplot(
        [max_mse_df.iloc[:, i] for i in range(n)],
        positions=pos,
        showmeans=True,
        showextrema=False,
        widths=0.6,
    )
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.7)
    vp["cmeans"].set_color(color)
    vp["cmeans"].set_linewidth(2.5)

    ax.errorbar(
        pos,
        means,
        yerr=yerr,
        linestyle="none",
        capsize=8,
        ecolor=color,
    )

    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [
            r"$\mathsf{RF}$",
            r"$\mathsf{MaggingRF}$",
            r"$\mathsf{MinMaxRF-M0}$",
            r"$\mathsf{MinMaxRF-M1}$",
            r"$\mathsf{MinMaxRF-M2}$",
            r"$\mathsf{MinMaxRF-M3}$",
            r"$\mathsf{MinMaxRF-M4}$",
        ]
    )
    ax.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )
    plt.tight_layout()
    if saveplot:
        os.makedirs("plots", exist_ok=True)
        outpath = os.path.join("plots", f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_envwise_mse(
    df: pd.DataFrame,
    saveplot=False,
    nameplot="mse_envs",
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
    ax.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.subplots_adjust(top=0.88, bottom=0.25)
    plt.tight_layout()
    if saveplot:
        os.makedirs("plots", exist_ok=True)
        outpath = os.path.join("plots", f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
