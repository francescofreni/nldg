import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAME_RF = "MaxRM-RF"
WIDTH, HEIGHT = 10, 6
# QUADRANTS = ["SW", "SE", "NW", "NE"]
QUADRANTS = ["Env 1", "Env 2", "Env 3", "Env 4"]


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
    color = "#5790FC"
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
        medianprops=dict(color="#5790FC", linewidth=2),
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
    out_dir: str | None = None,
) -> None:
    models = ["RF", f"{NAME_RF}({method})"]

    # Colors and offsets
    if method == "mse":
        colors = ["#5790FC", "#F89C20"]
    elif method == "nrw":
        colors = ["#5790FC", "#964A8B"]
    else:
        colors = ["#5790FC", "#E42536"]
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
        lab = "MSPE"
    elif method == "nrw":
        # lab = "Negative Reward"
        lab = "MSPE"
    else:
        # lab = "Regret"
        lab = "MSPE"
    ax.set_ylabel(lab)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)

    plt.tight_layout()
    if saveplot:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def plot_test_risk_all_methods(
    df: pd.DataFrame,
    saveplot: bool = False,
    nameplot: str = "heldout_mse_all_methods",
    show: bool = False,
    out_dir: str | None = None,
) -> None:
    models = [
        "RF",
        f"{NAME_RF}(mse)",
        f"{NAME_RF}(nrw)",
        f"{NAME_RF}(reg)",
    ]

    colors = {
        "RF": "#5790FC",
        f"{NAME_RF}(mse)": "#F89C20",
        f"{NAME_RF}(nrw)": "#964A8B",
        f"{NAME_RF}(reg)": "#E42536",
    }
    delta = 0.1
    offsets = np.linspace(-delta, delta, len(models))

    # Compute group stats
    grp = df.groupby(["HeldOutQuadrant", "Model"])["Test_risk"]
    means = grp.mean().unstack().reindex(QUADRANTS)
    stds = grp.std().unstack().reindex(QUADRANTS)
    counts = grp.count().unstack().reindex(QUADRANTS)
    ci95 = 1.96 * stds / np.sqrt(counts)

    x0 = np.arange(len(QUADRANTS))

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (off, model) in enumerate(zip(offsets, models)):
        xm = x0 + off
        ax.errorbar(
            xm,
            means[model],
            yerr=ci95[model],
            fmt="o",
            color=colors[model],
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
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)
    plt.tight_layout()

    if saveplot and out_dir:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def table_test_risk_all_methods(df: pd.DataFrame) -> pd.DataFrame:
    models = [
        "RF",
        f"{NAME_RF}(mse)",
        f"{NAME_RF}(nrw)",
        f"{NAME_RF}(reg)",
    ]

    # Compute stats
    grp = df.groupby(["HeldOutQuadrant", "Model"])["Test_risk"]
    means = grp.mean().unstack().reindex(QUADRANTS)
    stds = grp.std().unstack().reindex(QUADRANTS)
    counts = grp.count().unstack().reindex(QUADRANTS)
    ci95 = 1.96 * stds / np.sqrt(counts)

    # Combine into "mean ± CI" strings
    table_df = pd.DataFrame(index=QUADRANTS)
    for model in models:
        table_df[model] = [
            f"${means.loc[q, model]:.3f} \pm {ci95.loc[q, model]:.3f}$"
            for q in QUADRANTS
        ]

    # Reset index to have "Quadrant" as first column
    table_df = table_df.reset_index().rename(columns={"index": "Quadrant"})
    return table_df


def plot_envs_risk(
    df_env_spec: pd.DataFrame,
    df_main: pd.DataFrame | None = None,
    saveplot: bool = False,
    nameplot: str = "env_specific_mse",
    show: bool = False,
    method: str = "mse",
    out_dir: str | None = None,
):
    # Dynamically get model list
    models = ["RF", f"{NAME_RF}({method})"]
    num_models = len(models)

    # Colors and offsets
    if method == "mse":
        colors = ["#5790FC", "#F89C20"]
    elif method == "nrw":
        colors = ["#5790FC", "#964A8B"]
    else:
        colors = ["#5790FC", "#E42536"]
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
            env,
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
            ho,
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
        lab = "Negative Reward"
    else:
        lab = "Regret"
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


def plot_envs_mse_all_methods(
    df_env_all,
    saveplot=False,
    nameplot="env_specific_mse_all_methods",
    show=False,
    out_dir=None,
):
    df = (
        df_env_all.sort_values("Model")
        .drop_duplicates(
            subset=["HeldOutQuadrant", "EnvIndex", "Rep", "Model"]
        )
        .copy()
    )

    models = [
        "RF",
        f"{NAME_RF}(mse)",
        f"{NAME_RF}(nrw)",
        f"{NAME_RF}(reg)",
    ]
    colors = {
        "RF": "#5790FC",
        "MaxRM-RF(mse)": "#F89C20",
        "MaxRM-RF(nrw)": "#964A8B",
        "MaxRM-RF(reg)": "#E42536",
    }
    num_models = len(models)
    delta = 0.18
    n_subenv = 3
    figsize = (15, 6)

    fig, ax = plt.subplots(figsize=figsize)
    seen = {m: False for m in models}
    label_positions = []

    for i, ho in enumerate(QUADRANTS):
        subenvs = [q for q in QUADRANTS if q != ho]
        for j, env_name in enumerate(subenvs):
            x_base = i * n_subenv + j
            label_positions.append((ho, env_name, x_base))

            for m_idx, model in enumerate(models):
                ser = df[
                    (df["HeldOutQuadrant"] == ho)
                    & (df["Model"] == model)
                    & (df["EnvIndex"] == QUADRANTS.index(env_name))
                ]["MSE"]

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
                    color=colors[model],
                    markersize=8,
                    elinewidth=2.5,
                    capsize=0,
                    label=label,
                )

    # dashed separators between sub-environments
    for i in range(len(QUADRANTS)):  # for each held-out env block
        for j in range(1, n_subenv):  # between its sub-environments
            sep_x = i * n_subenv + j - 0.5
            ax.axvline(
                sep_x, linewidth=0.5, color="black", linestyle="--", alpha=0.5
            )

    # vertical separators
    for k in range(1, len(QUADRANTS)):
        sep_x = k * n_subenv - 0.5
        ax.axvline(sep_x, linewidth=0.5, color="black")

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(QUADRANTS) * n_subenv - 0.5)

    # label rows
    fig.canvas.draw()
    y0, y1 = ax.get_ylim()
    for _, env, x in label_positions:
        ax.text(
            x, y0 - 0.02 * (y1 - y0), env, ha="center", va="top", fontsize=10
        )

    # big labels per held-out env
    for i, ho in enumerate(QUADRANTS):
        mid = i * n_subenv + (n_subenv - 1) / 2
        ax.text(
            mid,
            y0 - 0.08 * (y1 - y0),
            ho,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel(r"$\mathsf{MSE}$")
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=14)

    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.tight_layout()

    if saveplot and out_dir is not None:
        plt.savefig(
            os.path.join(out_dir, f"{nameplot}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    if show:
        plt.show()


def plot_max_risk_vs_hyperparam(
    res: pd.DataFrame,
    hyperparam: str,
    saveplot: bool = False,
    nameplot: str = "max_mse_mtry",
    show: bool = False,
    out_dir: str | None = None,
    suffix: str = "mse",
) -> None:
    fontsize = 16
    methods = ["RF", f"{NAME_RF}({suffix})"]
    if suffix == "mse":
        colors = ["#5790FC", "#F89C20"]
    elif suffix == "nrw":
        colors = ["#5790FC", "#964A8B"]
    else:
        colors = ["#5790FC", "#E42536"]

    stats = []
    for method in methods:
        df_m = res[res["method"] == method]
        for v in sorted(df_m[hyperparam].unique()):
            vals = df_m[df_m[hyperparam] == v]["risk"].to_numpy()
            n = vals.size
            if n == 0:
                continue
            mean = vals.mean()
            stderr = vals.std(ddof=1) / np.sqrt(n)
            width = 1.96 * stderr
            stats.append(
                {
                    "method": method,
                    hyperparam: v,
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
            df_m[hyperparam],
            df_m["mean"],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            markeredgecolor="white",
        )
        plt.fill_between(
            df_m[hyperparam],
            df_m["lower"],
            df_m["upper"],
            color=color,
            alpha=0.3,
        )

    if hyperparam == "mtry":
        plt.xlabel(r"$m_{\mathrm{try}}$", fontsize=fontsize)
    elif hyperparam == "min_samples_leaf":
        plt.xlabel(
            "Minimum number of observations per leaf", fontsize=fontsize
        )
    else:
        plt.xlabel("Maximum depth", fontsize=fontsize)
    if suffix == "mse":
        lab = "MSE"
    elif suffix == "nrw":
        lab = "negative reward\n"
    else:
        lab = "regret"
    plt.ylabel(f"Maximum {lab} across environments", fontsize=fontsize)
    plt.grid(True, linewidth=0.2)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)

    # Set ticks exactly at the data points
    all_x_values = sorted(df_stats[hyperparam].unique())
    plt.xticks(all_x_values)

    plt.legend(frameon=True, fontsize=fontsize)
    plt.tight_layout()

    if saveplot and out_dir is not None:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


# def plot_max_mse_mtry(
#     res: pd.DataFrame,
#     saveplot: bool = False,
#     nameplot: str = "max_mse_mtry",
#     show: bool = False,
#     out_dir: str | None = None,
#     suffix: str = "mse",
# ) -> None:
#     methods = ["RF", f"{NAME_RF}({suffix})"]
#     if suffix == "mse":
#         colors = ["#5790FC", "#F89C20"]
#     elif suffix == "nrw":
#         colors = ["#5790FC", "#964A8B"]
#     else:
#         colors = ["#5790FC", "#E42536"]
#
#     nsim = res.groupby(["method", "mtry"]).size().iloc[0]
#
#     stats = []
#     for method in methods:
#         df_m = res[res["method"] == method]
#         for m in sorted(df_m["mtry"].unique()):
#             vals = df_m[df_m["mtry"] == m]["risk"]
#             mean = vals.mean()
#             stderr = vals.std(ddof=1) / np.sqrt(nsim)
#             width = 1.96 * stderr
#             stats.append(
#                 {
#                     "method": method,
#                     "mtry": m,
#                     "mean": mean,
#                     "lower": mean - width,
#                     "upper": mean + width,
#                 }
#             )
#
#     df_stats = pd.DataFrame(stats)
#
#     plt.figure(figsize=(8, 5))
#     for method, color in zip(methods, colors):
#         df_m = df_stats[df_stats["method"] == method]
#         plt.plot(
#             df_m["mtry"],
#             df_m["mean"],
#             label=method,
#             color=color,
#             marker="o",
#             linestyle="-",
#             markeredgecolor="white",
#         )
#         plt.fill_between(
#             df_m["mtry"],
#             df_m["lower"],
#             df_m["upper"],
#             color=color,
#             alpha=0.3,
#         )
#
#     plt.xlabel(r"$m_{\mathrm{try}}$")
#     if suffix == "mse":
#         lab = "MSE"
#     elif suffix == "nrw":
#         lab = "Negative Reward"
#     else:
#         lab = "Regret"
#     plt.ylabel(f"Maximum {lab} over environments")
#     plt.grid(True, linewidth=0.2)
#     plt.legend(frameon=True, fontsize=14)
#     plt.tight_layout()
#
#     if saveplot:
#         outpath = os.path.join(out_dir, f"{nameplot}.png")
#         plt.savefig(outpath, dpi=300, bbox_inches="tight")
#
#     if show:
#         plt.show()


def _ci95(std: pd.Series, n: pd.Series) -> pd.Series:
    return 1.96 * std / np.sqrt(n)


def write_lr_test_table_txt(
    main_df: pd.DataFrame,
    out_dir: str,
):
    """
    Creates a text file for the LR results.
    Column shows Test MSE mean ± 95% CI per held-out environment.
    """
    os.makedirs(out_dir, exist_ok=True)

    grp = main_df.groupby("HeldOutQuadrant")["Test_mse"]
    means = grp.mean().reindex(QUADRANTS)
    stds = grp.std().reindex(QUADRANTS)
    ns = grp.count().reindex(QUADRANTS)
    ci = _ci95(stds, ns)

    header = ["Quadrant", "LR"]
    lines = ["\t".join(header)]
    for q in QUADRANTS:
        m = means.loc[q]
        c = ci.loc[q]
        cell = f"{m:.3f} ± {c:.3f}"
        lines.append(f"{q}\t{cell}")

    with open(os.path.join(out_dir, "lr_test_mse.txt"), "w") as f:
        f.write("\n".join(lines))


def write_lr_env_specific_table_txt(
    env_metrics_df: pd.DataFrame,
    out_dir: str,
):
    """
    For each held-out environment, prints the MSE (mean ± 95% CI) achieved on the *training*
    environments during validation. Columns are the three training envs (the held-out env is omitted).
    """
    os.makedirs(out_dir, exist_ok=True)

    # stats per (heldout_idx, train_env_idx)
    grp = env_metrics_df.groupby(["HeldOutQuadrantIdx", "EnvIndex"])["MSE"]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])

    blocks = []
    for heldout_idx in range(len(QUADRANTS)):
        train_env_indices = [
            j for j in range(len(QUADRANTS)) if j != heldout_idx
        ]
        header = ["Held-out Env"] + [QUADRANTS[j] for j in train_env_indices]
        rows = ["\t".join(header)]

        row = [QUADRANTS[heldout_idx]]
        for j in train_env_indices:
            sub = stats[
                (stats["HeldOutQuadrantIdx"] == heldout_idx)
                & (stats["EnvIndex"] == j)
            ]
            m = float(sub["mean"].iloc[0])
            n = int(sub["count"].iloc[0])
            s = (
                float(sub["std"].iloc[0])
                if not np.isnan(sub["std"].iloc[0])
                else 0.0
            )
            c = (1.96 * s / np.sqrt(n)) if n > 1 else 0.0
            cell = f"{m:.3f} ± {c:.3f}"
            row.append(cell)

        rows.append("\t".join(row))
        blocks.append("\n".join(rows))

    content = ("\n\n").join(blocks) + "\n"
    with open(os.path.join(out_dir, "lr_val_envs_mse.txt"), "w") as f:
        f.write(content)


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
