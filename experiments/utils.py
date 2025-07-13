import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAME_RF = "WORME-RF"
WIDTH, HEIGHT = 10, 6


# =====================================
# Plotting functions for the simulation
# =====================================
def plot_max_mse_boxplot(
    max_mse_df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_boxplot",
    show: bool = False,
    out_dir: str | None = None,
) -> None:
    color = "tab:blue"
    n = len(max_mse_df.columns)
    pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(WIDTH + 4, HEIGHT))

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

    ax.set_ylabel("MSE")
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [
            "RF",
            "RF(magging)",
            f"{NAME_RF}(local)",
            f"{NAME_RF}(posthoc)",
            f"{NAME_RF}(posthoc-local)",
            f"{NAME_RF}(global-dfs)",
            f"{NAME_RF}(global)",
        ],
    )
    ax.grid(True, linewidth=0.2, axis="y")
    plt.tight_layout()

    if saveplot:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


# ============================================
# Plotting functions for the real data example
# ============================================
def plot_test_risk(
    df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "heldout_mse",
    show: bool = False,
    method: str = "mse",
    legend_pos: None | str = "lower left",
    out_dir: str | None = None,
) -> None:
    QUADRANTS = ["SW", "SE", "NW", "NE"]
    models = ["RF", f"{NAME_RF}(posthoc-{method})"]

    # Colors and offsets
    colors = ["lightskyblue", "orange"]
    delta = 0.1
    offsets = [-delta, delta]

    # Compute group stats
    grp = df.groupby(["HeldOutQuadrant", "Model"])["Test_risk"]
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
    if method == "mse":
        lab = r"$\mathsf{MSPE}$"
    elif method == "nrw":
        lab = r"$\mathsf{Negative Reward}$"
    else:
        lab = r"$\mathsf{Regret}$"
    ax.set_ylabel(lab)
    ax.legend(loc=legend_pos, frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)

    plt.tight_layout()
    if saveplot:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def plot_envs_risk(
    df_env_spec: pd.DataFrame,
    df_main: pd.DataFrame | None = None,
    saveplot: bool = False,
    nameplot: str = "env_specific_mse",
    show: bool = False,
    method: str = "mse",
    out_dir: str | None = None,
):
    QUADRANTS = ["SW", "SE", "NW", "NE"]

    # Dynamically get model list
    models = ["RF", f"{NAME_RF}(posthoc-{method})"]
    num_models = len(models)

    # Colors and offsets
    colors = ["lightskyblue", "orange"]
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
                ]["Risk"]

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
    if method == "mse":
        for i, ho in enumerate(QUADRANTS):
            for m_idx, model in enumerate(models):
                df_sub = df_main[
                    (df_main["HeldOutQuadrant"] == ho)
                    & (df_main["Model"] == model)
                ]

                if df_sub.empty:
                    continue

                train_mean = df_sub["Train_risk"].mean()
                train_std = df_sub["Train_risk"].std(ddof=1)
                ci95 = 1.96 * train_std / np.sqrt(len(df_sub))

                # Compute horizontal span range for this quadrant
                start = i * n_subenv - 0.4
                end = (i + 1) * n_subenv - 0.6

                label = (
                    f"{model} Pooled MSE"
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
    if method == "mse":
        lab = r"$\mathsf{MSE}$"
    elif method == "nrw":
        lab = r"$\mathsf{Negative Reward}$"
    else:
        lab = r"$\mathsf{Regret}$"
    ax.set_ylabel(lab)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)

    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.tight_layout()

    if saveplot:
        plt.savefig(
            os.path.join(out_dir, f"{nameplot}.png"),
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()


def plot_max_mse_mtry(
    res: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "max_mse_mtry",
    show: bool = False,
    out_dir: str | None = None,
    suffix: str = "mse",
) -> None:
    methods = ["RF", f"{NAME_RF}(posthoc-{suffix})"]
    colors = ["lightskyblue", "orange"]

    nsim = res.groupby(["method", "mtry"]).size().iloc[0]

    stats = []
    for method in methods:
        df_m = res[res["method"] == method]
        for m in sorted(df_m["mtry"].unique()):
            vals = df_m[df_m["mtry"] == m]["risk"]
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
    if suffix == "mse":
        lab = "MSE"
    elif suffix == "nrw":
        lab = "Negative Reward"
    else:
        lab = "Regret"
    plt.ylabel(f"Maximum {lab} over environments")
    plt.grid(True, linewidth=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()

    if saveplot:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


# ==========
# Additional
# ==========
def get_df(res_dict):
    rows = []
    for method, lists in res_dict.items():
        for sim_id, res_envs in enumerate(lists):
            for env_id, res in enumerate(res_envs):
                rows.append(
                    {
                        "MSE": res,
                        "env_id": env_id,
                        "sim_id": sim_id,
                        "method": method,
                    }
                )
    return pd.DataFrame(rows)
