import colorsys
import contextily as ctx
import copy
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import binomtest
from shapely.geometry import Point
from tqdm import tqdm
from typing import Tuple, Dict

N_ESTIMATORS = 25
MIN_SAMPLES_LEAF = 30
SEED = 42
N_JOBS = 5


#########################################################################
# Data loading and preprocessing functions
#########################################################################


def load_or_compute(filepath, compute_fn, args=None, rerun=False):
    """Load results from file or compute if needed."""
    if not rerun:
        if isinstance(filepath, list):
            if all(os.path.exists(fp) for fp in filepath):
                return [pd.read_csv(fp) for fp in filepath]
        elif os.path.exists(filepath):
            return pd.read_csv(filepath)

    results = compute_fn(**args)
    if isinstance(results, list) or isinstance(results, tuple):
        for i, res in enumerate(results):
            res.to_csv(filepath[i], index=False)
    else:
        results.to_csv(filepath, index=False)
    return results


def load_data(
    data_dir: str,
    env_function,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and preprocess the California housing dataset.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        Z (pd.DataFrame): Additional covariates
    """
    data_path = os.path.join(data_dir, "housing-sklearn.csv")
    dat = pd.read_csv(filepath_or_buffer=data_path)

    y = dat["MedHouseVal"]
    Z = dat[["Latitude", "Longitude"]]
    X = dat.drop(["MedHouseVal", "Latitude", "Longitude"], axis=1)

    env = env_function(Z, data_dir)
    if "DROP" in env:
        to_drop = env == "DROP"
        X = X.loc[~to_drop].reset_index(drop=True)
        y = y.loc[~to_drop].reset_index(drop=True)
        Z = Z.loc[~to_drop].reset_index(drop=True)
        env = env[~to_drop]
        env = pd.Categorical(env)
        unique_envs = env.categories
        env = env.codes

    return X, y, Z, env, unique_envs


def assign_county(Z: pd.DataFrame, data_dir: str) -> np.ndarray:
    """
    Creates the environment label based on county criteria.

    Args:
        Z (pd.DataFrame): Additional covariates (Latitude, Longitude)

    Returns:
        env (np.ndarray): Environment label
    """
    gdf = gpd.GeoDataFrame(
        Z,
        geometry=gpd.points_from_xy(Z["Longitude"], Z["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=4269)
    counties = gpd.read_file(
        os.path.join(
            data_dir, "cb_2024_us_county_500k/cb_2024_us_county_500k.shp"
        )
    )
    ca_counties = counties[counties["STATEFP"] == "06"]

    # Convert data to GeoDataFrame
    geometry = [Point(xy) for xy in zip(Z["Longitude"], Z["Latitude"])]

    # Spatial join to assign counties
    gdf = gpd.sjoin(
        gdf, ca_counties[["geometry", "NAME"]], how="left", predicate="within"
    )
    env = np.zeros(len(Z), dtype=object)
    value_counts = gdf["NAME"].value_counts()
    valid_counties = value_counts[value_counts >= 100].index.tolist()[:25]
    idx = gdf["NAME"].isin(valid_counties)
    env[idx] = gdf.loc[idx, "NAME"]
    env[~idx] = "DROP"

    return env


def assign_quadrant(
    Z: pd.DataFrame,
) -> np.ndarray:
    """
    Creates the environment label based on geographic criteria.

    Args:
        Z (pd.DataFrame): Additional covariates (Latitude, Longitude)

    Returns:
        env (np.ndarray): Environment label
    """
    lat, lon = Z["Latitude"], Z["Longitude"]

    # north = lat >= 35
    # south = ~north
    # east = lon >= -120
    # west = ~east
    #
    # env = np.zeros(len(Z), dtype=int)
    # env[south & west] = 0  # SW
    # env[south & east] = 1  # SE
    # env[north & west] = 2  # NW
    # env[north & east] = 3  # NE

    west = lon < -121.5
    east = ~west
    sw = (lat < 38) & west
    nw = (lat >= 38) & west
    lat_thr = 34.5
    se = (lat < lat_thr) & east
    ne = (lat >= lat_thr) & east

    env = np.zeros(len(Z), dtype=int)
    env[sw] = 0  # SW
    env[se] = 1  # SE
    env[nw] = 2  # NW
    env[ne] = 3  # NE

    return env


#########################################################################
# Model modification functions
#########################################################################


def modify_rf(rf, risk, Ytr, Etr, Xte, fitted_erm=None, fitted_erm_trees=None):
    solvers = ["CLARABEL", "ECOS", "SCS"]
    success = False
    kwargs = {"n_jobs": N_JOBS}
    rf_maxrm = copy.deepcopy(rf)

    if risk == "regret":
        kwargs["sols_erm"] = fitted_erm
        kwargs["sols_erm_trees"] = fitted_erm_trees

    for solver in solvers:
        try:
            rf_maxrm.modify_predictions_trees(
                Etr,
                method=risk,
                **kwargs,
                solver=solver,
            )
            success = True
            break
        except Exception as e_try:
            pass
    if not success:
        rf_maxrm.modify_predictions_trees(
            Etr,
            method=risk,
            **kwargs,
            opt_method="extragradient",
        )
    return rf_maxrm


#########################################################################
# Plotting functions
#########################################################################


def make_5_colors_hls(base_color):
    r, g, b = mcolors.to_rgb(base_color)

    # convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # generate increasing lightness but keep hue + saturation
    lightness_values = np.linspace(max(0, l * 0.8), min(1, l * 1.4), 5)

    colors = []
    for L in lightness_values:
        R, G, B = colorsys.hls_to_rgb(h, L, s)
        colors.append((R, G, B))

    return colors


def plot_test_risk_all_methods(
    df: pd.DataFrame,
    models: list[str],
    colors: dict[str, str],
    folds: bool = False,
    saveplot: bool = False,
    nameplot: str = "heldout_mse_all_methods",
    show: bool = False,
    out_dir: str | None = None,
) -> None:
    delta = 0.1
    offsets = np.linspace(-delta, delta, len(models))

    # Compute group stats
    grp = df.groupby(["HeldOut"])
    means = grp.mean().unstack()
    stds = grp.std().unstack()
    counts = grp.count().unstack()
    ci95 = 1.96 * stds / np.sqrt(counts)

    held_out_sets = df["HeldOut"].unique()
    x0 = np.arange(len(held_out_sets))

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
    if folds:
        heldOut = [f"Fold {i}" for i in range(1, len(held_out_sets) + 1)]
        for i, q in enumerate(held_out_sets):
            print(f"Fold {i + 1}: {q}")
        ax.set_xticklabels(heldOut)
    else:
        ax.set_xticklabels(held_out_sets)
    ax.set_xlabel("Held-Out Environment")
    ax.set_ylabel(r"$\mathsf{MSPE}$")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.7)
    plt.tight_layout()

    if saveplot and out_dir:
        outpath = os.path.join(out_dir, f"{nameplot}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_env_with_basemap(
    env, Z, out_dir, label="Quadrant", counties=None, y=None, clustering=None
):
    df = Z.copy()
    df[label] = env
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    num_envs = 4 if label == "Quadrant" else len(counties)

    if label == "Quadrant":
        if y is None:
            raise ValueError("y must be provided when label is 'Quadrant'")
        colors = {0: "#5790FC", 1: "#F89C20", 2: "#964A8B", 3: "#E42536"}
        labels = {
            0: rf"Env 1: $\bar{{y}}$ = {round(np.mean(y[env == 0]), 2)}",
            1: rf"Env 2: $\bar{{y}}$ = {round(np.mean(y[env == 1]), 2)}",
            2: rf"Env 3: $\bar{{y}}$ = {round(np.mean(y[env == 2]), 2)}",
            3: rf"Env 4: $\bar{{y}}$ = {round(np.mean(y[env == 3]), 2)}",
        }

    elif label == "County":
        if clustering is None:
            colors_list = sns.color_palette("hsv", len(counties)).as_hex()
            colors_list = [
                "#e6194b",
                "#3cb44b",
                "#ffe119",
                "#0082c8",
                "#f58231",
                "#449cbf",
                "#46f0f0",
                "#f032e6",
                "#d2f53c",
                "#eb6565",
                "#008080",
                "#e6beff",
                "#aa6e28",
                "#eb974e",
                "#800000",
                "#aaffc3",
                "#808000",
                "#e78c8c",
                "#000080",
                "#808080",
                "#e2ec27",
                "#DD9748",
                "#baffc9",
                "#92eb5e",
                "#7b389f",
            ]
            colors = {i: colors_list[i] for i in range(len(counties))}
            labels = {i: counties[i] for i in range(len(counties))}
        else:
            for i, cli in enumerate(clustering):
                for j, clj in enumerate(clustering):
                    if i != j:
                        assert not set(cli).intersection(
                            set(clj)
                        ), f"Clusters {i} and {j} overlap!"

            g1, g2, g3, g4, g5 = clustering
            ordering = g1 + g2 + g3 + g4 + g5
            base_colors = [
                "#F89C20",
                "#4FB793",
                "#5790FC",
                "#964A8B",
                "#3AC3E5",
            ]
            c1, c2, c3, c4, c5 = [make_5_colors_hls(b) for b in base_colors]
            county_to_color = {}
            for i, county in enumerate(ordering):
                if county in g1:
                    color = c1[g1.index(county)]
                elif county in g2:
                    color = c2[g2.index(county)]
                elif county in g3:
                    color = c3[g3.index(county)]
                elif county in g4:
                    color = c4[g4.index(county)]
                elif county in g5:
                    color = c5[g5.index(county)]
                county_to_color[county] = color

            colors = {
                i: county_to_color[counties[i]] for i in range(len(counties))
            }
            labels = {i: counties[i] for i in range(len(counties))}
    else:
        raise ValueError(f"Unknown label: {label}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for q in range(num_envs):
        gdf[gdf[label] == q].plot(
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
        for q in range(num_envs)
    ]
    if label == "County" and clustering is not None:
        left_folds = {
            "Fold I": g1,
            "Fold II": g2,
        }

        right_folds = {
            "Fold III": g3,
            "Fold IV": g4,
            "Fold V": g5,
        }

        def build_legend_elements(folds):
            elements = []
            for fold_name, group_list in folds.items():
                # Fold subheading
                elements.append(Line2D([], [], color="none", label=fold_name))

                # Counties under the fold
                for county in group_list:
                    i = counties.get_loc(county)
                    elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            label=labels[i],
                            markerfacecolor=colors[i],
                            markersize=10,
                            alpha=0.8,
                        )
                    )
            return elements

        left_elements = build_legend_elements(left_folds)
        right_elements = build_legend_elements(right_folds)

        # Create the left legend
        leg1 = ax.legend(
            handles=left_elements,
            title="",
            loc="upper left",
            bbox_to_anchor=(0.46, 1.0),
            frameon=False,
        )
        ax.add_artist(leg1)

        # Create the right legend
        leg2 = ax.legend(
            handles=right_elements,
            loc="upper left",
            bbox_to_anchor=(0.72, 1.0),
            frameon=False,
        )
        for legend in [leg1, leg2]:
            for text in legend.get_texts():
                if text.get_text().startswith("Fold"):
                    text.set_weight("bold")

    else:
        ax.legend(handles=legend_elements, title=label, loc="best")

    ctx.add_basemap(
        ax, source=ctx.providers.CartoDB.Positron, attribution=False
    )
    ax.set_axis_off()
    plt.tight_layout()
    outpath = os.path.join(out_dir, f"ca_housing_envs_{label}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_results(
    results_df,
    env,
    unique_envs,
    figsize,
    filepath,
    add_n=False,
    annot=False,
    clustering=None,
):
    # Reorder based on clustering if provided
    if clustering is not None:
        ordered_envs = [env for cluster in clustering for env in cluster]
        heatmap_data = results_df.pivot(
            index="TrainEnv", columns="TestEnv", values="MSE"
        )
        heatmap_data = heatmap_data.reindex(
            index=ordered_envs, columns=ordered_envs
        )
    else:
        heatmap_data = results_df.pivot(
            index="TrainEnv", columns="TestEnv", values="MSE"
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"size": 7},
        ax=ax,
    )
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("MSE", rotation=90, labelpad=15)

    # Add clustering visualization
    if clustering is not None:
        pos = 0
        n_envs = len(heatmap_data)
        roman = ["I", "II", "III", "IV", "V"]
        for i, cluster in enumerate(clustering):
            cluster_size = len(cluster)
            ax.plot(
                [pos, pos + cluster_size],
                [-0.5, -0.5],
                "k-",
                linewidth=1,
                clip_on=False,
            )
            ax.plot([pos, pos], [-0.5, -0.3], "k-", linewidth=1, clip_on=False)
            ax.plot(
                [pos + cluster_size, pos + cluster_size],
                [-0.5, -0.3],
                "k-",
                linewidth=1,
                clip_on=False,
            )
            ax.text(
                pos + cluster_size / 2,
                -1.2,
                f"Fold {roman[i]}",
                ha="center",
                va="center",
            )

            pos += cluster_size

    ax.set_xlabel("Test Environment")
    ax.set_ylabel("Train Environment")
    if add_n:
        num_per_env = np.unique(env, return_counts=True)
        ax.set_yticks(np.arange(len(unique_envs)) + 0.5)
        ax.set_yticklabels(
            [
                f"{q}\n(n={num_per_env[1][i]})"
                for i, q in enumerate(unique_envs)
            ],
            rotation=0,
        )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return heatmap_data


def plot_oob_mse(
    heatmap_data, unique_envs, filename, clustering=None, fold_size=5
):
    diag_data = heatmap_data.values.diagonal()

    if clustering is None:
        np.random.seed(SEED)
        counties_shuffled = np.random.permutation(unique_envs)
        counties_folds = [
            counties_shuffled[i : i + fold_size]
            for i in range(0, len(unique_envs), fold_size)
        ]
    else:
        counties_folds = clustering

    county_fold_dict = {}
    for fold_idx, fold in enumerate(counties_folds):
        for county in fold:
            county_fold_dict[county] = fold_idx

    hue = [county_fold_dict[county] for county in unique_envs]

    df = pd.DataFrame(
        {
            "County": unique_envs,
            "OOB_MSE": diag_data,
            "Fold": hue,
        }
    )

    if clustering is not None:
        ordering = [e for cluster in clustering for e in cluster]
        idx_ordering = [list(unique_envs).index(e) for e in ordering]
        df["order"] = ordering
    else:
        ordering = None

    plt.figure(figsize=(8, 2.5))
    sns.barplot(
        data=df,
        x="County",
        y="OOB_MSE",
        order=ordering,
        hue="Fold",
        palette=sns.color_palette("tab10", len(counties_folds)),
    )
    plt.ylabel("OOB MSE")
    plt.xlabel("County")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calibration_plot(preds_test_df, held_out, models, colors, filepath):
    ymin = min(preds_test_df["true"].min(), preds_test_df["predicted"].min())
    ymax = max(preds_test_df["true"].max(), preds_test_df["predicted"].max())
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models:
        df_preds = preds_test_df[
            (preds_test_df["HeldOut"] == held_out)
            & (preds_test_df["Model"] == model)
        ]
        sns.regplot(
            x="true",
            y="predicted",
            data=df_preds,
            scatter=True,
            label=f"{model}",
            color=colors[model],
            ax=ax,
            scatter_kws={"alpha": 0.2, "s": 2},
        )
    ax.plot([ymin, ymax], [ymin, ymax], color="k", linestyle="--")
    ax.set_xlabel("True Median House Price")
    ax.set_ylabel(f"Predicted Median House Price")
    ax.set_title(f"Calibration Plot\nHeld-Out: {held_out}")
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])

    leg = ax.legend()
    for text in leg.get_texts():
        label = text.get_text()
        text_color = colors[label]
        text.set_color(text_color)

    plt.tight_layout()
    plot_path = os.path.join(filepath)
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_diff_in_max_mse(results_df, filename):
    results_test_df_agg = (
        results_df.groupby(["HeldOut", "fold_split"])
        .agg("max")
        .drop(columns=["EnvIndex"])
        .reset_index()
    )
    diff = results_test_df_agg["MaxRM-RF(mse)"] - results_test_df_agg["RF"]
    diff = diff / results_test_df_agg["RF"] * 100
    plt.figure(figsize=(4, 3))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
    sns.histplot(diff, color="#F89C20", bins=50, kde=False, stat="count")
    plt.axvline(
        np.median(diff),
        color="red",
        linestyle="--",
        label=f"Median {np.median(diff):.2f}% improvement",
    )
    plt.xlim(-60, 60)
    plt.ylim(0, 33)
    plt.legend()
    plt.ylabel("Count")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel(
        "Relative difference in maximum MSE\nbetween maxRM-RF(mse) and RF"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


#########################################################################
# Analysis functions
#########################################################################


def permutation_test_max_mse(
    df_model1: pd.DataFrame,
    df_model2: pd.DataFrame,
    domain_col: str = "env_id",
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_seed: int = None,
) -> Dict:
    """
    Permutation test comparing max MSE between two models.

    Permutes predictions within each domain independently, then computes
    the difference in max MSE across domains.

    Parameters:
    -----------
    df_model1 : pd.DataFrame
        Predictions from first model (baseline)
        Must have columns: 'true', 'predicted', 'residual', 'domain_col'
    df_model2 : pd.DataFrame
        Predictions from second model
        Must have same structure and domains as df_model1
    domain_col : str
        Column name for domain identifier (default: 'env_id')
    n_permutations : int
        Number of permutations (default: 10000)
    alternative : str
        'two-sided': test if models differ
        'less': test if model2 has lower max MSE than model1
        'greater': test if model2 has higher max MSE than model1
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict with keys:
        - 'observed_diff': observed difference (max_MSE_model1 - max_MSE_model2)
        - 'p_value': permutation p-value
        - 'max_mse_model1': max MSE across domains for model1
        - 'max_mse_model2': max MSE across domains for model2
        - 'mse_per_domain_model1': MSE per domain for model1
        - 'mse_per_domain_model2': MSE per domain for model2
        - 'permutation_distribution': array of permuted differences
        - 'n_permutations': number of permutations
        - 'alternative': alternative hypothesis used
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Verify domains match
    domains_1 = set(df_model1[domain_col].unique())
    domains_2 = set(df_model2[domain_col].unique())

    if domains_1 != domains_2:
        raise ValueError(
            f"Domains don't match between models. "
            f"Model1: {domains_1}, Model2: {domains_2}"
        )

    domains = sorted(domains_1)

    # Verify same instances in each domain
    for domain in domains:
        n1 = len(df_model1[df_model1[domain_col] == domain])
        n2 = len(df_model2[df_model2[domain_col] == domain])
        if n1 != n2:
            raise ValueError(
                f"Domain {domain} has different number of instances:"
                f"model1={n1}, model2={n2}"
            )

    # Compute observed MSE per domain
    mse_per_domain_1 = df_model1.groupby(domain_col)["residual"].apply(
        lambda x: np.mean(x**2)
    )
    mse_per_domain_2 = df_model2.groupby(domain_col)["residual"].apply(
        lambda x: np.mean(x**2)
    )

    # Observed max MSE
    max_mse_1 = mse_per_domain_1.max()
    max_mse_2 = mse_per_domain_2.max()
    observed_diff = max_mse_1 - max_mse_2

    # Permutation test
    permuted_diffs = np.zeros(n_permutations)

    for perm_idx in range(n_permutations):
        mse_perm_1 = []
        mse_perm_2 = []

        # Permute within each domain independently
        for dom in domains:
            # Get residuals for this domain from both models
            res_1 = df_model1[df_model1[domain_col] == dom]["residual"].values
            res_2 = df_model2[df_model2[domain_col] == dom]["residual"].values

            # Permute residuals
            perm_res_1 = res_1.copy()
            perm_res_2 = res_2.copy()
            for i in range(len(res_1)):
                if np.random.rand() < 0.5:
                    # Swap residuals
                    perm_res_1[i], perm_res_2[i] = perm_res_2[i], perm_res_1[i]

            # Compute MSE for permuted data
            mse_perm_1.append(np.mean(perm_res_1**2))
            mse_perm_2.append(np.mean(perm_res_2**2))

        # Compute max MSE difference for this permutation
        permuted_diffs[perm_idx] = np.max(mse_perm_1) - np.max(mse_perm_2)

    # Compute p-value based on alternative hypothesis
    if alternative == "two-sided":
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    elif alternative == "less":
        # Test if model2 has significantly lower max MSE (observed_diff > 0)
        p_value = np.mean(permuted_diffs >= observed_diff)
    elif alternative == "greater":
        # Test if model2 has significantly higher max MSE (observed_diff < 0)
        p_value = np.mean(permuted_diffs <= observed_diff)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', got {alternative}"
        )

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "max_mse_model1": max_mse_1,
        "max_mse_model2": max_mse_2,
        "mse_per_domain_model1": mse_per_domain_1.to_dict(),
        "mse_per_domain_model2": mse_per_domain_2.to_dict(),
        "permutation_distribution": permuted_diffs,
        "n_permutations": n_permutations,
        "alternative": alternative,
    }


def table_test_risk_all_methods_perm(
    df: pd.DataFrame,
    df_preds: pd.DataFrame,
    models: list[str],
    folds: bool = False,
    perm: bool = False,
    return_p_values: bool = False,
) -> pd.DataFrame:
    means = df.copy().set_index("HeldOut").unstack()
    held_out_sets = df["HeldOut"].unique()

    # Combine into "mean" strings
    table_df = pd.DataFrame(index=held_out_sets)
    for model in models:
        table_df[model] = [
            f"${means.loc[(model, q)]:.3f}$" for q in held_out_sets
        ]

    # if the mean is the lowest per row, make it bold
    for q in held_out_sets:
        row_means = {model: means.loc[(model, q)] for model in models}
        min_model = min(row_means, key=row_means.get)
        cell = table_df.loc[q, min_model]
        table_df.loc[q, min_model] = f"\\bm{{{cell}}}"

    # Shade in grey the methods that are statistically better than RF
    if return_p_values:
        pval_df = pd.DataFrame(index=held_out_sets, columns=models)
    if perm:
        for q in tqdm(held_out_sets):
            held_out_idx = df_preds["HeldOut"] == q
            df_rf = df_preds[(df_preds["Model"] == "RF") & held_out_idx]
            for model in models:
                if model == "RF":
                    continue
                df_model = df_preds[
                    (df_preds["Model"] == model) & held_out_idx
                ]
                perm = permutation_test_max_mse(
                    df_rf,
                    df_model,
                    domain_col="domain_col",
                    n_permutations=10000,
                    alternative="less",
                    random_seed=42,
                )
                if return_p_values:
                    pval_df.loc[q, model] = perm["p_value"]
                # Bonferroni correction
                if perm["p_value"] < 0.05 / (
                    (len(models) - 1) * len(held_out_sets)
                ):
                    cell = table_df.loc[q, model]
                    table_df.loc[q, model] = f"\\cellcolor{{gray!25}} {cell}"

    # for each row, add a comment in the last cell with the index
    for q in held_out_sets:
        table_df.loc[q, models[-1]] += f" \\\\ % {q}"
    # add a column at the start called Fold
    table_df.insert(0, "Fold", ["Fold" for _ in range(len(held_out_sets))])

    if return_p_values:
        return table_df, pval_df
    else:
        return table_df


def print_worst_case_environments(results_df, unique_envs, models):
    print("--------------------------------")
    print("Worst-case environments per held-out set:")
    results_agg = results_df.groupby("HeldOut").max().reset_index()
    worst_env_df = pd.DataFrame(index=results_agg["HeldOut"])
    for model in models:
        wc_envs = []
        for held_out in worst_env_df.index:
            worst_env = results_df[(results_df["HeldOut"] == held_out)][
                model
            ].idxmax()
            wc_envs.append(unique_envs[results_df.loc[worst_env, "EnvIndex"]])
        worst_env_df[model] = wc_envs
    print(worst_env_df)


def generate_tables_and_plots(
    results_df,
    preds_df,
    agg_type,
    models,
    rf_models,
    colors,
    out_dir,
    prefix="",
):
    """Generate all tables and plots for a given aggregation type."""
    results_agg = (
        results_df.groupby(["HeldOut"])
        .agg(agg_type)
        .drop(columns=["EnvIndex"])
        .reset_index()
    )

    # Generate tables
    for model_set, suffix in [
        (models, "all_methods"),
        (
            ["LR", "GroupDRO-NN"],
            "only_mse_methods",
        ),
    ]:
        table_df = table_test_risk_all_methods_perm(
            results_agg,
            preds_df,
            model_set,
            folds=True,
            perm=False,
        )
        latex_str = table_df.to_latex(
            index=False, escape=False, column_format="l" + "c" * len(model_set)
        )
        filepath = os.path.join(
            out_dir, f"{prefix}l5co_{agg_type}_{suffix}.txt"
        )
        with open(filepath, "w") as f:
            f.write(latex_str)

    # Generate plots
    # plot_test_risk_all_methods(
    #     results_agg,
    #     models,
    #     colors,
    #     saveplot=True,
    #     out_dir=out_dir,
    #     nameplot=f"{prefix}l5co_{agg_type}_all_methods",
    #     folds=True,
    # )
    # plot_test_risk_all_methods(
    #     results_agg,
    #     rf_models,
    #     colors,
    #     saveplot=True,
    #     out_dir=out_dir,
    #     nameplot=f"{prefix}l5co_{agg_type}_rf_methods",
    #     folds=True,
    # )


def print_model_comparison_stats(results_df, model_names, baseline="RF"):
    """Print comprehensive comparison statistics for models vs baseline."""
    results_agg = (
        results_df.groupby(["HeldOut", "fold_split"])
        .agg("max")
        .drop(columns=["EnvIndex"])
        .reset_index()
    )

    for model in model_names:
        diffs = results_agg[model] - results_agg[baseline]
        c = np.sum(results_agg[model] <= results_agg[baseline])
        m, med = np.mean(diffs), np.median(diffs)

        print("--------------------------------")
        print(f"Results for {model}:")
        print(
            f"\t# folds where {model} <= {baseline}: {c} / {len(results_agg)}"
        )
        print(f"\tMean difference (Model - {baseline}): {m:.4f}")
        print(f"\tMedian difference (Model - {baseline}): {med:.4f}")
        print(f"\tMean {baseline}: {np.mean(results_agg[baseline]):.4f}")
        print(f"\tMean {model}: {np.mean(results_agg[model]):.4f}")

        # Confidence intervals
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
        print(f"\t95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ci_lower_norm = m - 1.96 * se
        ci_upper_norm = m + 1.96 * se
        print(
            f"\t95% CI (normal approx): [{ci_lower_norm:.4f}, {ci_upper_norm:.4f}]"
        )

        # Statistical tests
        t_stat, p_value = stats.ttest_rel(
            results_agg[model], results_agg[baseline]
        )
        print(f"\tPaired t-test p-value: {p_value:.4f}")

        p_binom = binomtest(
            k=np.sum(diffs < 0), n=len(diffs), p=0.5, alternative="greater"
        ).pvalue
        print(
            f"\tBinomial p-value for {model} better than {baseline}: {p_binom:.4f}"
        )
