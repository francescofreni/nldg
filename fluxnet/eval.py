# Code modified from https://github.com/anyafries/fluxnet_bench.git
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


def nse(ytrue, ypred):
    numerator = np.sum((ytrue - ypred) ** 2)
    denominator = np.sum((ytrue - np.mean(ytrue)) ** 2)
    return 1 - (numerator / denominator)


def evaluate_fold(ytrue, ypred, verbose=True, digits=3):
    mse = mean_squared_error(ytrue, ypred)
    results = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2_score": r2_score(ytrue, ypred),
        "relative_error": np.mean(np.abs(ytrue - ypred) / np.abs(ytrue)),
        "mae": np.mean(np.abs(ytrue - ypred)),
        "nse": nse(ytrue, ypred),
    }
    if verbose:
        logger.info(f"* RESULTS over {len(ytrue)} predictions:")
        logger.info(
            "\t "
            + ", ".join(
                f"{metric.upper()}={value:.{digits}f}"
                for metric, value in results.items()
            )
        )
    return results


def latex_table_best_per_group(
    df,
    metric,
    better="lower",
    column_order=["RF", "MaxRM-RF(mse)", "MaxRM-RF(nrw)", "MaxRM-RF(reg)"],
):
    metric_df = df[df["metric"] == metric].copy()
    table_df = metric_df.pivot(index="group", columns="model", values="value")

    if column_order:
        table_df = table_df.reindex(columns=column_order)

    abs_max = table_df.abs().max().max()
    scale_factor = None
    if abs_max < 1e-4:
        scale_factor = 10 ** int(-np.floor(np.log10(abs_max)))
        table_df *= scale_factor
    elif abs_max > 1e4:
        scale_factor = 10 ** int(np.floor(np.log10(abs_max)))
        table_df /= scale_factor

    if better == "lower":
        best_values = table_df.min(axis=1)
    else:
        best_values = table_df.max(axis=1)

    formatted_df = table_df.copy().astype(str)
    for group in table_df.index:
        for model in table_df.columns:
            val = table_df.loc[group, model]
            formatted_val = f"{val:.3g}"
            if val == best_values[group]:
                formatted_val = f"\\bm{{{formatted_val}}}"
            formatted_df.loc[group, model] = f"${formatted_val}$"

    n_methods = formatted_df.shape[1]
    column_format = "l" + "c" * (n_methods)
    formatted_df.index.name = None
    formatted_df.columns.name = "Site"
    latex_str = formatted_df.to_latex(
        escape=False, column_format=column_format
    )

    if scale_factor:
        latex_str += (
            f"\n% Note: Values scaled by 1e{int(np.log10(scale_factor))}"
        )

    return latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default=os.path.join(BASE_DIR, "data_cleaned")
    )
    parser.add_argument("--override", type=bool, default=False)
    parser.add_argument("--agg", type=str, default="daily-10")
    parser.add_argument("--setting", type=str, default="loso")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--metric", type=str, default="rmse")

    path = parser.parse_args().path
    override = parser.parse_args().override
    agg = parser.parse_args().agg
    setting = parser.parse_args().setting
    start = parser.parse_args().start
    stop = parser.parse_args().stop
    cv = parser.parse_args().cv
    metric = parser.parse_args().metric

    # Find any files of the form results/{agg}_{setting}*
    results_dir = os.path.join(BASE_DIR, "results")
    results_files = [
        f for f in os.listdir(results_dir) if f.startswith(f"{agg}_{setting}")
    ]  # need to add cv here

    # divide the results files into models
    groups = {}
    for filename in results_files:
        parts = filename.split("_")
        if len(parts) > 2:  # Ensure there are at least three parts
            model_name = parts[2]  # Extract the third part (model_name)
            groups.setdefault(model_name, []).append(filename)

    # Load data
    all_results = []
    for model_name, files in groups.items():
        model_data = []
        for f in files:
            df = pd.read_csv(os.path.join(results_dir, f))
            model_data.append(df)
        model_data = pd.concat(model_data, ignore_index=True)
        model_name = model_name.replace("posthoc-", "")
        model_data["model"] = model_name
        all_results.append(model_data)

    # Only select the first entry for each group (in case of multiple runs)
    all_results = pd.concat(all_results, ignore_index=True)

    # Keep only the groups that have been analysed with all models
    groups_per_model = all_results.groupby("model")["group"].apply(set)
    common_groups = set.intersection(*groups_per_model)
    all_results = all_results[all_results["group"].isin(common_groups)]

    all_results = all_results.groupby(
        ["model", "group"], as_index=False
    ).first()

    if setting in ["l10so", "l5so", "l3so"]:
        metrics = ["max_mse_test", "max_rmse_test"]
    else:
        metrics = ["mse", "rmse", "r2_score", "relative_error", "mae", "nse"]
    plot_df = all_results.melt(
        id_vars=["model", "group"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    if metric in [
        "mse",
        "rmse",
        "relative_error",
        "mae",
        "max_mse_test",
        "max_rmse_test",
    ]:
        better = "lower"
    else:
        better = "upper"
    print(latex_table_best_per_group(plot_df, metric=metric, better=better))

    # # Plot the results
    # sns.boxplot(x="model", y="rmse", data=all_results)
    # plt.ylim(0, 0.5)
    # plt.savefig("results/plots_tmp/boxplot_rmse.png")
    # plt.close()
    #
    # sns.boxplot(x="model", y="nse", data=all_results, showfliers=False)
    # # plt.ylim(-2, 1)
    # plt.savefig("results/plots_tmp/boxplot_nse.png")
    # plt.close()
    #
    # sns.boxplot(x="model", y="r2_score", data=all_results, showfliers=False)
    # plt.savefig("results/plots_tmp/boxplot_r2.png")
    # plt.close()
    #
    # sns.histplot(
    #     data=all_results, x="rmse", hue="model", kde=True, bins=20, alpha=0.4
    # )
    # plt.savefig("results/plots_tmp/histo_rmse.png")
    # plt.close()
