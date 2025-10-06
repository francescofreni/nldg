import os
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


def generate_fold_info(df, setting, fold_size=5, seed=42):
    if setting == "loso":
        sites = df["site_id"].dropna().unique()
        groups = sorted(sites)

    elif setting == "logo":
        dataset = df.copy()
        dataset = dataset.dropna(axis=1, how="all")
        dataset = dataset.dropna(axis=0, how="any")

        lat = np.radians(dataset["latitude"])
        lon = np.radians(dataset["longitude"])
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        coords = np.vstack((x, y, z)).T
        kmeans = KMeans(n_clusters=4, random_state=0).fit(coords)
        labels = kmeans.labels_
        unique_labels = np.unique(labels)
        groups = []
        for lab in unique_labels:
            groups.append(np.unique(dataset[labels == lab]["site_id"]))

    elif setting == "l5so":
        sites = pd.Series(df["site_id"].dropna().unique())
        sites = sites.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )
        groups = [
            list(sites[i : i + fold_size])
            for i in range(0, len(sites), fold_size)
        ]

    return groups


def get_fold_df(
    df,
    setting,
    group,
    target="GPP",
    cv=False,
    remove_missing=False,
    seed=42,
):
    df_out = df.copy()

    # drop columns
    df_out.drop(
        columns=["time", "longitude", "latitude", "year"], inplace=True
    )
    if "date" in df.columns:
        df_out.drop(columns="date", inplace=True)

    df_out["season"] = df_out["season"].astype(int)
    df_out = pd.get_dummies(
        df_out, columns=["season"], prefix="season", dtype=np.float64
    )
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
    df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24)
    df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24)
    df_out.drop(columns=["month", "hour"], inplace=True)

    # drop missing
    if any(df_out.isna().mean() == 1):
        logger.warning(
            f"Column `{df_out.columns[df_out.isna().mean() == 1][0]}` is missing for group {group}: it is being dropped"
        )
        df_out = df_out.dropna(axis=1, how="all")
    if remove_missing:
        nstart = df_out.shape[0]
        df_out = df_out.dropna(axis=0, how="any")
        nout = df_out.shape[0]
        if nstart > nout:
            logger.info(
                f"* Dropped {nstart-nout}/{nstart} ({(nstart-nout)/nstart * 100:.2f}%) rows due to missingness"
            )

    # split into train/test
    if setting == "loso":
        # split it by site
        train = df_out.loc[df_out["site_id"] != group].copy()
        test = df_out.loc[df_out["site_id"] == group].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            return None, None, None, None, None, None
    elif setting in ["l5so", "logo"]:
        train = df_out.loc[~df_out["site_id"].isin(group)].copy()
        test = df_out.loc[df_out["site_id"].isin(group)].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            return None, None, None, None, None, None
    del df_out

    # drop outliers
    q1, q99 = train["GPP"].quantile([0.01, 0.99])
    ndrop = np.mean((train["GPP"] < q1) | (train["GPP"] > q99))
    logger.info(f"* Dropping {ndrop*100:.2f}% training outliers")
    train = train.loc[(train["GPP"] > q1) & (train["GPP"] < q99)]
    ndrop_test = np.mean((test["GPP"] < q1) | (test["GPP"] > q99))
    logger.info(f"* Dropping {ndrop_test*100:.2f}% test outliers")
    test = test.loc[(test["GPP"] > q1) & (test["GPP"] < q99)]

    # clean up
    train_ids = train["site_id"]
    test_ids = test["site_id"].copy()
    train.drop(columns="site_id", inplace=True)
    test.drop(columns="site_id", inplace=True)

    xcols = train.columns != target
    ycol = train.columns == target

    # split into x,y
    xtrain, ytrain = train.values[:, xcols], train.values[:, ycol].ravel()
    xtest, ytest = test.values[:, xcols], test.values[:, ycol].ravel()

    if cv:
        env = np.asarray(train_ids)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_folds = [
            (tr_idx, va_idx) for tr_idx, va_idx in skf.split(xtrain, env)
        ]
        return xtrain, ytrain, xtest, ytest, train_ids, test_ids, cv_folds
    else:
        return xtrain, ytrain, xtest, ytest, train_ids, test_ids
