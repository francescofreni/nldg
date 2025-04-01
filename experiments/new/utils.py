import os
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


def plot_mse_r2(
    mse_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
    out: bool = True,
    isd: bool = False,
    nn: bool = False,
) -> None:
    """
    Plots the MSE and R2 comparison between different methods.

    Args:
         mse_df: DataFrame containing the MSE values for the methods.
         r2_df: DataFrame containing the R2 values for the methods.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
         out: If true, the results refer to the test data.
         isd: If true, include the results of Invariant Subspace Decomposition.
         nn: True if the data is relative to the neural networks example.
    """
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    if isd:
        n_methods = 5
    elif nn:
        n_methods = 3
    else:
        n_methods = 4

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 2, HEIGHT))
    if out:
        fig.suptitle(
            r"$\mathsf{MSE}$ and $R^2$ comparison - Test data",
            fontsize=22,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            r"$\mathsf{MSE}$ and $R^2$ comparison - Train data",
            fontsize=22,
            fontweight="bold",
        )

    # Create violin plots
    vp_mse = ax[0].violinplot(
        [mse_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    vp_r2 = ax[1].violinplot(
        [r2_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors for MSE violins
    for i, vp in enumerate(vp_mse["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Set colors for R² violins
    for i, vp in enumerate(vp_r2["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Set colors for mean lines
    vp_mse["cmedians"].set_color(c)
    vp_mse["cmedians"].set_linewidth(2.5)

    vp_r2["cmedians"].set_color(c)
    vp_r2["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax[0].set_ylabel(r"$\mathsf{MSE}$")
    ax[1].set_ylabel(r"$R^2$")

    ax[0].set_xticks(range(n_methods))
    ax[1].set_xticks(range(n_methods))

    if isd:
        labels = [
            r"$\mathsf{RF}$",
            r"$\mathsf{MaximinRF}$",
            r"$\mathsf{MaggingRF-Forest}$",
            r"$\mathsf{MaggingRF-Trees}$",
            r"$\mathsf{IsdRF}$",
        ]
    elif nn:
        labels = [
            r"$\mathsf{NN}$",
            r"$\mathsf{MaximinNN}$",
            r"$\mathsf{MaggingNN}$",
        ]
    else:
        labels = [
            r"$\mathsf{RF}$",
            r"$\mathsf{MaximinRF}$",
            r"$\mathsf{MaggingRF-Forest}$",
            r"$\mathsf{MaggingRF-Trees}$",
        ]
    ax[0].set_xticklabels(labels)
    ax[1].set_xticklabels(labels)

    ax[0].grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )
    ax[1].grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_minxplvar(
    maxmse_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
    isd: bool = False,
    nn: bool = False,
) -> None:
    """
    Plots the maximum MSE across environments to compare different methods.

    Args:
         maxmse_df: DataFrame containing the maximum MSE values
            for the methods.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
         isd: If true, include the results of Invariant Subspace Decomposition.
         nn: True if the data is relative to the neural networks example.
    """
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    if isd:
        n_methods = 5
    elif nn:
        n_methods = 3
    else:
        n_methods = 4

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.suptitle(
        r"Minimal explained variance across environments",
        fontsize=22,
        fontweight="bold",
    )

    vp_minxplvar = ax.violinplot(
        [maxmse_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors for the violins
    for i, vp in enumerate(vp_minxplvar["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Customize the mean markers
    vp_minxplvar["cmedians"].set_color(c)
    vp_minxplvar["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.set_xticks(range(n_methods))
    if isd:
        ax.set_xticklabels(
            [
                r"$\mathsf{RF}$",
                r"$\mathsf{MaximinRF}$",
                r"$\mathsf{MaggingRF-Forest}$",
                r"$\mathsf{MaggingRF-Trees}$",
                r"$\mathsf{IsdRF}$",
            ]
        )
    elif nn:
        ax.set_xticklabels(
            [
                r"$\mathsf{NN}$",
                r"$\mathsf{MaximinNN}$",
                r"$\mathsf{MaggingNN}$",
            ]
        )
    else:
        ax.set_xticklabels(
            [
                r"$\mathsf{RF}$",
                r"$\mathsf{MaximinRF}$",
                r"$\mathsf{MaggingRF-Forest}$",
                r"$\mathsf{MaggingRF-Trees}$",
            ]
        )

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

    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_maxmse(
    maxmse_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
    isd: bool = False,
    nn: bool = False,
) -> None:
    """
    Plots the maximum MSE across environments to compare different methods.

    Args:
         maxmse_df: DataFrame containing the maximum MSE values
            for the methods.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
         isd: If true, include the results of Invariant Subspace Decomposition.
         nn: True if the data is relative to the neural networks example.
    """
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    if isd:
        n_methods = 5
    elif nn:
        n_methods = 3
    else:
        n_methods = 4

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.suptitle(
        r"Maximum MSE across environments",
        fontsize=22,
        fontweight="bold",
    )

    vp_maxmse = ax.violinplot(
        [maxmse_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors for the violins
    for i, vp in enumerate(vp_maxmse["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Customize the mean markers
    vp_maxmse["cmedians"].set_color(c)
    vp_maxmse["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.set_xticks(range(n_methods))
    if isd:
        ax.set_xticklabels(
            [
                r"$\mathsf{RF}$",
                r"$\mathsf{MaximinRF}$",
                r"$\mathsf{MaggingRF-Forest}$",
                r"$\mathsf{MaggingRF-Trees}$",
                r"$\mathsf{IsdRF}$",
            ]
        )
    elif nn:
        ax.set_xticklabels(
            [
                r"$\mathsf{NN}$",
                r"$\mathsf{MaximinNN}$",
                r"$\mathsf{MaggingNN}$",
            ]
        )
    else:
        ax.set_xticklabels(
            [
                r"$\mathsf{RF}$",
                r"$\mathsf{MaximinRF}$",
                r"$\mathsf{MaggingRF-Forest}$",
                r"$\mathsf{MaggingRF-Trees}$",
            ]
        )

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

    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_invrec(
    invrec_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
) -> None:
    """
    Plots the maximum MSE across environments to compare different methods.

    Args:
         invrec_df: DataFrame containing the MSE comparing the fitted values and the true response values.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
    """
    c = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
    n_methods = 5

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    vp_invrec = ax.violinplot(
        [invrec_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors for the violins
    for i, vp in enumerate(vp_invrec["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Customize the mean markers
    vp_invrec["cmedians"].set_color(c)
    vp_invrec["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(
        [
            r"$\mathsf{RF}$",
            r"$\mathsf{MaximinRF}$",
            r"$\mathsf{MaggingRF-Forest}$",
            r"$\mathsf{MaggingRF-Trees}$",
            r"$\mathsf{IsdRF}$",
        ]
    )

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

    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_weights_magging(
    maxmse_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
) -> None:
    """
    Plots the weights used for magging.

    Args:
         maxmse_df: DataFrame containing the maximum MSE values
            for the methods.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
    """
    c = ["tab:blue", "tab:orange", "tab:green"]
    n_envs = 3

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.suptitle(
        r"Weights used in $\mathsf{MaggingRF}$", fontsize=22, fontweight="bold"
    )

    vp_w = ax.violinplot(
        [maxmse_df.iloc[:, i] for i in range(n_envs)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_envs),
    )

    # Set colors for the violins
    for i, vp in enumerate(vp_w["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Customize the mean markers
    vp_w["cmedians"].set_color(c)
    vp_w["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax.set_xticks(range(n_envs))
    ax.set_xticklabels([r"Env $1$", r"Env $2$", r"Env $3$"])

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

    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_mse_r2_adapt(
    mse_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    name_plot: str,
    plots_folder: str,
    out: bool = True,
) -> None:
    """
    Plots the MSE and R2 comparison between different methods.

    Args:
         mse_df: DataFrame containing the MSE values for the methods.
         r2_df: DataFrame containing the R2 values for the methods.
         name_plot: Name of the plot to save in the dedicated folder.
         plots_folder: Folder where to save the plots.
         out: If true, the results refer to the test data.
         isd: If true, include the results of Invariant Subspace Decomposition.
    """
    c = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:red",
        "tab:olive",
        "tab:cyan",
    ]
    n_methods = 7

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 2, HEIGHT))
    if out:
        fig.suptitle(
            r"$\mathsf{MSE}$ and $R^2$ comparison - Test data",
            fontsize=22,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            r"$\mathsf{MSE}$ and $R^2$ comparison - Train data",
            fontsize=22,
            fontweight="bold",
        )

    # Create violin plots
    vp_mse = ax[0].violinplot(
        [mse_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    vp_r2 = ax[1].violinplot(
        [r2_df.iloc[:, i] for i in range(n_methods)],
        showmedians=True,
        showextrema=False,
        widths=0.4,
        positions=range(n_methods),
    )

    # Set colors for MSE violins
    for i, vp in enumerate(vp_mse["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Set colors for R² violins
    for i, vp in enumerate(vp_r2["bodies"]):
        vp.set_facecolor(c[i])
        vp.set_edgecolor(c[i])
        vp.set_alpha(0.7)

    # Set colors for mean lines
    vp_mse["cmedians"].set_color(c)
    vp_mse["cmedians"].set_linewidth(2.5)

    vp_r2["cmedians"].set_color(c)
    vp_r2["cmedians"].set_linewidth(2.5)

    # Labels and formatting
    ax[0].set_ylabel(r"$\mathsf{MSE}$")
    ax[1].set_ylabel(r"$R^2$")

    ax[0].set_xticks(range(n_methods))
    ax[1].set_xticks(range(n_methods))

    labels = [
        r"$\mathsf{RF}$",
        r"$\mathsf{MaximinRF}$",
        r"$\mathsf{MaggingRF-Forest}$",
        r"$\mathsf{MaggingRF-Trees}$",
        r"$\mathsf{IsdRF}$",
        r"$\mathsf{IsdRFad}$",
        r"$\mathsf{RFad}$",
    ]

    ax[0].set_xticklabels(labels)
    ax[1].set_xticklabels(labels)

    ax[0].grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )
    ax[1].grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.3, alpha=0.3
    )

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plots_folder, name_plot)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
