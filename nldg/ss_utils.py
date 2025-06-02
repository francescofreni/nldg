import numpy as np
import matplotlib.pyplot as plt


def plot_dtr(
    dtr,
    x_grid,
    preds_erm,
    preds_maximin,
    preds_magging=None,
    optfun=None,
    saveplot=False,
    nameplot="setting5",
):
    line_colors = ["lightskyblue", "orange", "mediumpurple"]
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

    ax.plot(x_grid, preds_erm, color=line_colors[0], linewidth=2, label="SS")
    ax.plot(
        x_grid,
        preds_maximin,
        color=line_colors[1],
        linewidth=2,
        label="MaximinSS",
    )
    if preds_magging is not None:
        ax.plot(
            x_grid,
            preds_magging,
            color=line_colors[2],
            linewidth=2,
            label="MaggingSS",
        )

    if optfun == 1:
        y_opt = 0.8 * np.sin(x_grid / 2) ** 2 + 3
        ax.plot(
            x_grid,
            y_opt,
            color="orangered",
            linewidth=2,
            linestyle="--",
            label="Optimal",
        )
    elif optfun == 2:
        y_opt = np.where(x_grid > 0, 2.4 * x_grid, -2.4 * x_grid)
        ax.plot(
            x_grid,
            y_opt,
            color="orangered",
            linewidth=2,
            linestyle="--",
            label="Optimal",
        )
    elif optfun == 3:
        y_opt = np.where(x_grid > 0, 1.86 * x_grid, 1.63 * x_grid)
        ax.plot(
            x_grid,
            y_opt,
            color="orangered",
            linewidth=2,
            linestyle="--",
            label="Optimal",
        )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    if saveplot:
        plt.savefig(f"{nameplot}.png", dpi=300, bbox_inches="tight")
    plt.show()
