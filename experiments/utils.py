import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 16,
        "axes.unicode_minus": True,
    }
)
WIDTH = 12.0
HEIGHT = 6.0


def plot_mse(
    mse_df: pd.DataFrame,
    out: bool = True,
) -> None:
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    n_methods = len(mse_df.columns)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    title = (
        r"$\mathsf{MSPE}$ comparison" if out else r"$\mathsf{MSE}$ comparison"
    )
    fig.suptitle(
        title,
        fontsize=22,
        fontweight="bold",
    )

    # Create violin plot
    vp = ax.violinplot(
        [mse_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors
    for i, vp_body in enumerate(vp["bodies"]):
        vp_body.set_facecolor(c[i])
        vp_body.set_edgecolor(c[i])
        vp_body.set_alpha(0.7)

    vp["cmedians"].set_color(c[:n_methods])
    vp["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ylab = r"$\mathsf{MSPE}$" if out else r"$\mathsf{MSE}$"
    ax.set_ylabel(ylab)
    ax.set_xticks(range(n_methods))
    labels = [
        r"$\mathsf{RF}$",
        r"$\mathsf{MaximinRF-Local}$",
        r"$\mathsf{MaximinRF-Global}$",
        r"$\mathsf{MaggingRF}$",
    ]
    ax.set_xticklabels(labels)

    ax.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def plot_time_maxmse_minxv(
    df: pd.DataFrame,
    comparison_metric: int = 1,
) -> None:
    c = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
    ]
    n_methods = len(df.columns)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    if comparison_metric == 1:
        title = r"Runtime comparison"
    elif comparison_metric == 2:
        title = r"Maximum $\mathsf{MSE}$ over training environments"
    else:
        title = r"Minimal explained variance over training environments"
    fig.suptitle(
        title,
        fontsize=22,
        fontweight="bold",
    )

    # Create violin plot
    vp = ax.violinplot(
        [df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors
    for i, vp_body in enumerate(vp["bodies"]):
        vp_body.set_facecolor(c[i])
        vp_body.set_edgecolor(c[i])
        vp_body.set_alpha(0.7)

    vp["cmedians"].set_color(c[:n_methods])
    vp["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    if comparison_metric == 1:
        ylab = r"Time (seconds)"
    elif comparison_metric == 2:
        ylab = r"$\mathsf{MSE}$"
    else:
        ylab = r"Explained variance"
    ax.set_ylabel(ylab)
    ax.set_xticks(range(n_methods))
    labels = [
        r"$\mathsf{RF}$",
        r"$\mathsf{MaximinRF-Local}$",
        r"$\mathsf{MaximinRF-Global}$",
        r"$\mathsf{MaggingRF}$",
    ]
    ax.set_xticklabels(labels)

    ax.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def plot_weights_magging(
    df: pd.DataFrame,
) -> None:
    n_envs = len(df.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        r"Weights used in $\mathsf{MaggingRF}$", fontsize=22, fontweight="bold"
    )

    vp_w = ax.violinplot(
        [df.iloc[:, i] for i in range(n_envs)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_envs),
    )

    # Set colors for the violins
    for i, vp in enumerate(vp_w["bodies"]):
        vp.set_facecolor("tab:blue")
        vp.set_edgecolor("tab:blue")
        vp.set_alpha(0.7)

    # Customize the mean markers
    vp_w["cmedians"].set_color(["tab:blue"] * n_envs)
    vp_w["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax.set_xticks(range(n_envs))
    ax.set_xticklabels([r"Env $1$", r"Env $2$"])

    ax.grid(
        True,
        which="both",
        axis="y",
        color="grey",
        linestyle="--",
        linewidth=0.3,
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


def plot_envwise_metric(df: pd.DataFrame, mse=True) -> None:
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
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
    title = (
        r"$\mathsf{MSE}$ by training environment"
        if mse
        else r"Explained variance by training environment"
    )
    fig.suptitle(title, fontsize=22, fontweight="bold")

    group_spacing = n_envs
    method_tick_positions = []
    method_tick_labels = []
    violin_positions = []

    # Plot violins
    for i, method in enumerate(methods):
        for j, env_id in enumerate(envs):
            pos = i * group_spacing + j
            violin_positions.append((method, env_id, pos))

            metric = "MSE" if mse else "xplvar"
            data = df[(df["method"] == method) & (df["env_id"] == env_id)][
                metric
            ]
            vp = ax.violinplot(
                data,
                positions=[pos],
                showmedians=True,
                showextrema=False,
                widths=0.7,
            )

            color = c[i % len(c)]
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.7)

            vp["cmedians"].set_color(c[i])
            vp["cmedians"].set_linewidth(2)

        # Track center of method group for x-tick label
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

    ylab = r"$\mathsf{MSE}$" if mse else r"Explained variance"
    ax.set_ylabel(ylab)
    ax.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.subplots_adjust(top=0.88, bottom=0.25)
    plt.tight_layout()
    plt.show()
