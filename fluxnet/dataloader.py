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


def generate_fold_info(df, setting, start=0, stop=None, fold_size=5, seed=42):
    if setting in ["insite", "insite-random", "loso"]:
        sites = df["site_id"].dropna().unique()
        if setting in ["insite", "insite-random"]:
            # only keep sites with at least 8 years of data
            site_years = df.groupby("site_id")["year"].nunique()
            sites = site_years[site_years >= 8].index.values
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

    elif setting == "l5so":
        sites = pd.Series(df["site_id"].dropna().unique())
        sites = sites.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )
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
    num=False,
    min_samples=None,
):
    # Get the correct data
    if setting == "insite" or setting == "insite-random":
        df_out = df.loc[df["site_id"] == group].copy()
    elif setting in ["logo", "loso", "l5so"]:
        df_out = df.copy()

    # drop rows where target is missing
    df_out = df_out.dropna(subset=[target])

    # create time features
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

    # drop columns, but keep year info for splitting
    df_out.drop(
        columns=["time", "longitude", "latitude"], inplace=True
    )
    if "date" in df.columns:
        df_out.drop(columns="date", inplace=True)

    # split into train/test
    if setting == "insite":
        # split it chronologically
        site_years = df_out["year"].value_counts().sort_index()
        site_years = site_years.index[site_years >= min_samples]
        n_years = len(site_years)
        if n_years < 8:
            logger.warning(
                f"* SKIPPING {group}: only {n_years} years with >= {min_samples} samples"
            )
            return None, None, None, None, None, None
        unique_years = np.sort(site_years)
        train_years, test_years = unique_years[:4], unique_years[4:8]
        train = df_out.loc[df_out["year"].isin(train_years)].copy()
        test = df_out.loc[df_out["year"].isin(test_years)].copy()
    elif setting == "insite-random":
        # split it randomly
        raise NotImplementedError("insite-random not implemented yet")
        unique_years = np.sort(df_out["year"].unique())
        df_out = df_out.loc[df_out["year"].isin(unique_years[:8])].copy()
        n_train = int(df_out.shape[0] * 0.5)
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
    elif setting in ["l5so", "logo"]:
        train = df_out.loc[~df_out["site_id"].isin(group)].copy()
        test = df_out.loc[df_out["site_id"].isin(group)].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            return None, None, None, None, None, None
    del df_out

    # # drop outliers
    # q1, q99 = train[target].quantile([0.01, 0.99])
    # ndrop = np.mean((train[target] < q1) | (train[target] > q99))
    # logger.info(f"* Dropping {ndrop*100:.2f}% training outliers")
    # train = train.loc[(train[target] > q1) & (train[target] < q99)]
    # ndrop_test = np.mean((test[target] < q1) | (test[target] > q99))
    # logger.info(f"* Dropping {ndrop_test*100:.2f}% test outliers")
    # test = test.loc[(test[target] > q1) & (test[target] < q99)]

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
    env_col = "year" if setting in ["insite", "insite-random"] else "site_id"
    train_ids = train[env_col]
    test_ids = test[env_col].copy()
    train = train.drop(columns=["site_id", "year"]).astype(np.float64)
    test = test.drop(columns=["site_id", "year"]).astype(np.float64) 

    xcols = ~train.columns.isin(['GPP', 'NEE'])
    ycol = train.columns == target

    # split into x,y
    xtrain, ytrain = train.values[:, xcols], train.values[:, ycol].ravel()
    xtest, ytest = test.values[:, xcols], test.values[:, ycol].ravel()

    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)

    if num:
        exclude_cols = [
            target,
            "IGBP_veg_MF",
            "IGBP_veg_GRA",
            "IGBP_veg_ENF",
            "IGBP_veg_SAV",
            "IGBP_veg_EBF",
            "IGBP_veg_WSA",
            "IGBP_veg_DBF",
            "IGBP_veg_OSH",
            "IGBP_veg_CRO",
            "IGBP_veg_CSH",
            "IGBP_veg_WET",
            "IGBP_veg_CVM",
            "season_1",
            "season_2",
            "season_3",
            "season_4",
        ]
        xcols = ~train.columns.isin(exclude_cols)
        xtrain_num = train.loc[:, xcols].values
        xtest_num = test.loc[:, xcols].values
        return (
            xtrain,
            ytrain,
            xtest,
            ytest,
            train_ids,
            test_ids,
            xtrain_num,
            xtest_num,
        )
    else:
        return xtrain, ytrain, xtest, ytest, train_ids, test_ids
