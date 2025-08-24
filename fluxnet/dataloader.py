# Code modified from https://github.com/anyafries/fluxnet_bench.git
import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


# def generate_fold_info(df, setting, start, stop=None):
#     if setting in ["insite", "insite-random", "loso"]:
#         sites = df["site_id"].unique()
#         if stop is None:
#             stop = len(sites)
#         groups = sorted(sites)[start:stop]
#     elif setting == "logo":
#         if stop is None or stop > 10:
#             stop = 10
#         groups = range(start, stop)
#     return groups


def generate_fold_info(df, setting, start=0, stop=None, fold_size=10):
    if setting in ["insite", "insite-random", "loso"]:
        sites = df["site_id"].dropna().unique()
        sites = sorted(sites)
        if stop is None:
            stop = len(sites)
        groups = sites[start:stop]

    elif setting == "logo":
        # if stop is None or stop > 10:
        #     stop = 10
        # groups = list(range(start, stop))
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

    elif (
        setting == "l10so"
        or setting == "l5so"
        or setting == "l3so"
        or setting == "l15so"
    ):
        sites = pd.Series(df["site_id"].dropna().unique())
        sites = sites.sample(frac=1.0, random_state=42).reset_index(drop=True)
        folds = [
            list(sites[i : i + fold_size])
            for i in range(0, len(sites), fold_size)
        ]
        groups = folds[slice(start, stop)]

    return groups


def get_fold_df(
    df,
    setting,
    group,
    target="GPP",
    cv=False,
    remove_missing=False,
    astorch=False,
):
    # Get the correct data
    if setting == "insite" or setting == "insite-random":
        df_out = df.loc[df["site_id"] == group].copy()
    elif setting in ["logo", "loso", "l10so", "l5so", "l3so", "l15so"]:
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
    if setting == "insite":
        # split it chronologically
        n_train = int(df_out.shape[0] * 0.8)
        train = df_out.iloc[:n_train]
        test = df_out.iloc[n_train:]
    elif setting == "insite-random":
        # split it randomly
        n_train = int(df_out.shape[0] * 0.8)
        train = df_out.sample(n=n_train, random_state=1)
        test = df_out.drop(train.index)
    # elif setting == "logo":
    #     data_path = os.path.join(BASE_DIR, "data_cleaned")
    #     sites = pd.read_csv(os.path.join(data_path, "grouping_equal_size.csv"))
    #     sites = sites.loc[sites["balanced_cluster"] == group, "site"]
    #     train = df_out.loc[~df_out["site_id"].isin(sites)].copy()
    #     test = df_out.loc[df_out["site_id"].isin(sites)].copy()
    elif setting == "loso":
        # split it by site
        train = df_out.loc[df_out["site_id"] != group].copy()
        test = df_out.loc[df_out["site_id"] == group].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            return None, None, None, None, None, None
    elif setting in ["l10so", "l5so", "l3so", "logo", "l15so"]:
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

    # # Standardization
    # to_standardize = [
    #     'Tair', 'vpd', 'SWdown', 'LWdown', 'SWdown_clearsky',
    #     'LST_TERRA_Day', 'LST_TERRA_Night', 'EVI', 'NIRv',
    #     'NDWI_band7', 'LAI', 'fPAR'
    # ]
    # to_standardize = [c for c in to_standardize if c in train.columns]
    #
    # # Fit global scalers on TRAIN only
    # x_scaler = StandardScaler().fit(train[to_standardize])
    # y_scaler = StandardScaler().fit(train[[target]])
    #
    # # Apply to TRAIN and TEST
    # train.loc[:, to_standardize] = x_scaler.transform(train[to_standardize])
    # test.loc[:, to_standardize] = x_scaler.transform(test[to_standardize])
    #
    # train.loc[:, [target]] = y_scaler.transform(train[[target]])
    # test.loc[:, [target]] = y_scaler.transform(test[[target]])

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

    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)

    return xtrain, ytrain, xtest, ytest, train_ids, test_ids
