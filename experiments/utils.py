import os
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    'axes.labelsize': 22,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 16,
    'axes.unicode_minus': True,
})
WIDTH = 12.0
HEIGHT = 6.0


def plot_mse_r2(
    mse_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    name_plot: str,
    name_method: str = 'maximin',
) -> None:
    """
    Plots the MSE and R2 comparison between two different methods.

    Args:
         mse_df: Dataframe containing the MSE values for the two methods.
         r2_df: Dataframe containing the R2 values for the two methods.
         name_plot: Name of the plot to save in the dedicated folder.
         name_method: Name of the method to compare against the default Random Forest.
            Accepted values are 'maximin' and 'isd'.
    """
    c = ['tab:blue', 'tab:orange']
    vp_mse = [None] * 2
    vp_r2 = [None] * 2
    axs0 = [None] * 2
    axs1 = [None] * 2
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))

    for i in range(2):
        if i == 0:
            axs0[0] = ax[0]
            axs1[0] = ax[1]
        else:
            axs0[i] = ax[0].twinx()
            axs1[i] = ax[1].twinx()
            axs0[i].set_yticks([])
            axs1[i].set_yticks([])
        vp_mse[i] = axs0[i].violinplot(mse_df.iloc[:, i], showmeans=True, showextrema=False,
                                       widths=0.5,
                                       positions=[i])
        vp_r2[i] = axs1[i].violinplot(r2_df.iloc[:, i], showmeans=True, showextrema=False,
                                      widths=0.5,
                                      positions=[i])

    for i in range(2):
        for vp in vp_mse[i]['bodies']:
            vp.set_color(c[i])
        for vp in vp_r2[i]['bodies']:
            vp.set_color(c[i])
        vp_mse[i]['cmeans'].set_color(c[i])
        vp_mse[i]['cmeans'].set_linewidth(2.5)
        vp_r2[i]['cmeans'].set_color(c[i])
        vp_r2[i]['cmeans'].set_linewidth(2.5)

    # ax[0].set_title("MSE Comparison", fontsize=12)
    # ax[1].set_title("R-squared Comparison", fontsize=12)
    ax[0].set_ylabel(r'$\mathsf{MSE}$')
    ax[1].set_ylabel(r'$R^2$')

    ax[0].set_xticks([0, 1])
    ax[1].set_xticks([0, 1])
    if name_method == 'maximin':
        ax[0].set_xticklabels([r'$\mathsf{RF}$', r'$\mathsf{MaximinRF}$'])
        ax[1].set_xticklabels([r'$\mathsf{RF}$', r'$\mathsf{MaximinRF}$'])
    else:
        ax[0].set_xticklabels([r'$\mathsf{RF}$', r'$\mathsf{IsdRF}$'])
        ax[1].set_xticklabels([r'$\mathsf{RF}$', r'$\mathsf{IsdRF}$'])

    for i in range(2):
        axs0[i].grid(True, which='both', axis='y', color='grey',
                     linestyle='--', linewidth=0.3, alpha=0.3)
        axs1[i].grid(True, which='both', axis='y', color='grey',
                     linestyle='--', linewidth=0.3, alpha=0.3)

    # ax[0].legend([vp_mse[0]['cmeans'], vp_mse[1]['cmeans'],
    #              vp_r2[0]['cmeans'], vp_r2[1]['cmeans']],
    #             [r'$RF$', r'$InfRF$'], loc='upper left', ncol=2)

    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, name_plot)

    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
