import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from adaXT.random_forest import RandomForest
from nldg.additional.data import DataContainer
from nldg.additional.gdro import GroupDRO
from nldg.rf import MaggingRF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ca_housing_utils import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_ca_housing")
os.makedirs(OUT_DIR, exist_ok=True)
# OUT_DIR = os.path.join(OUT_DIR, "1203")
# os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use(os.path.join(SCRIPT_DIR, "style.mplstyle"))

# Experiment flags
DOMAINS = "counties"  
OUT_DIR = os.path.join(OUT_DIR, DOMAINS)
os.makedirs(OUT_DIR, exist_ok=True)

# Experiment parameters
B = 20
VAL_PERCENTAGE = 0.3
SEED = 42

# Geographical clusters of counties in California
G1 = ['Solano', 'Sonoma', 'Contra Costa', 'Marin', 'San Francisco']
G2 = ['San Joaquin','Sacramento', 'Butte', 'Placer', 'Stanislaus' ]
G3 = ['Alameda', 'Santa Clara', 'San Mateo', 'Monterey', 'Santa Cruz']
G4 = ['Fresno','Santa Barbara', 'Ventura', 'Tulare' ,'Kern']
G5 = ['Los Angeles', 'Orange', 'Riverside', 'San Bernardino', 'San Diego']

# Models and colors for plotting
models = [
    'LR', 'RF', 'GroupDRO-NN', 
    'Magging-RF(mse)', 'Magging-RF(nrw)', 'Magging-RF(reg)',
    'MaxRM-RF(mse)', 'MaxRM-RF(nrw)', 'MaxRM-RF(reg)'
]
rf_models = ["RF", "MaxRM-RF(mse)", "MaxRM-RF(nrw)", "MaxRM-RF(reg)"]
colors = {
    "LR": "#4FB793",
    "RF": "#5790FC",
    "MaxRM-RF(mse)": "#F89C20",
    "MaxRM-RF(nrw)": "#F24C00",
    "MaxRM-RF(reg)": "#B40426",
    "GroupDRO-NN": "#86C8DD",
    'GroupDRO-NN2': "#45A3C6",
    'GroupDRO-NN3': "#1D6F8E",
    "Magging-RF(mse)": "#C55CB6",
    "Magging-RF(nrw)": "#964A8B",
    "Magging-RF(reg)": "#6F3666",
}

#########################################################################
# Experiment functions
#########################################################################

def similarity_experiment(unique_envs, X, y, env):
    results = []
    n_envs = len(unique_envs)
    for qi_train, qi_test in tqdm(
        [(i, j) for i in range(n_envs) for j in range(n_envs)]
    ):
        train_mask = env == qi_train
        X_tr, Y_tr = X[train_mask], y[train_mask]
        env_tr = env[train_mask]
        
        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
            sampling_args={'OOB':True} if qi_train == qi_test else None,
        )
        rf.fit(X_tr, Y_tr)

        if qi_train == qi_test:
            mse_rf = rf.oob
        else:
            test_mask = env == qi_test
            X_te, Y_te = X[test_mask], y[test_mask]
            env_te = env[test_mask]
            pred_rf = rf.predict(X_te)
            mse_rf = mean_squared_error(Y_te, pred_rf)
        
        results.append({
            "TrainEnv": unique_envs[qi_train],
            "TestEnv": unique_envs[qi_test],
            "MSE": mse_rf,
        })
    results_df = pd.DataFrame(results)
    return results_df


def run_experiment(Xtr, Ytr, Etr, Xval, Yval, Eval, Xte, Yte, Ete, data,
                   verbose=False):

    # Get solutions of individual ERM in each environment ------------
    fitted_erm = np.zeros(len(Etr))
    fitted_erm_trees = np.zeros((N_ESTIMATORS, len(Etr)))
    pred_erm = np.zeros(len(Ete))
    for env in np.unique(Etr):
        mask = Etr == env
        Xtr_env = Xtr[mask]
        Ytr_env = Ytr[mask]
        rf_e = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=SEED,
            n_jobs=N_JOBS,
        )
        rf_e.fit(Xtr_env, Ytr_env)
        fitted_erm[mask] = rf_e.predict(Xtr_env)
        for i in range(N_ESTIMATORS):
            fitted_erm_trees[i, mask] = rf_e.trees[i].predict(
                np.ascontiguousarray(Xtr_env)
            )
        mask_te = Ete == env
        pred_erm[mask_te] = rf_e.predict(Xte[mask_te])

    # LR ------------------------------------------------------------
    if verbose: print("Fitting Linear Regression")
    lr = LinearRegression()
    lr.fit(Xtr, Ytr)
    # ---------------------------------------------------------------

    # RF ------------------------------------------------------------
    if verbose: print("Fitting Random Forest")
    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=SEED,
        n_jobs=N_JOBS,
    )
    rf.fit(Xtr, Ytr)
    # ---------------------------------------------------------------

    # Magging -------------------------------------------------------
    if verbose: print("Fitting Magging-RF(mse)")
    rf_magging_mse = MaggingRF(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        backend="adaXT",
        risk="mse", # negative reward is the default for magging
        # sols_erm=fitted_erm,
    )
    rf_magging_mse.fit(np.array(Xtr), np.array(Ytr), np.array(Etr))
    # ---------------------------------------------------------------

    # Magging -------------------------------------------------------
    if verbose: print("Fitting Magging-RF(reg)")
    rf_magging_reg = MaggingRF(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        backend="adaXT",
        risk="reg", # negative reward is the default for magging
        sols_erm=fitted_erm,
    )
    rf_magging_reg.fit(np.array(Xtr), np.array(Ytr), np.array(Etr))
    # ---------------------------------------------------------------

    # Magging -------------------------------------------------------
    if verbose: print("Fitting Magging-RF(nrw)")
    rf_magging = MaggingRF(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        backend="adaXT",
        risk="nrw", # negative reward is the default for magging
        sols_erm=fitted_erm,
    )
    rf_magging.fit(Xtr, Ytr, Etr)
    # ---------------------------------------------------------------

    # GroupDRO-NN ---------------------------------------------------
    if verbose: print("Fitting GroupDRO-NN")
    gdro = GroupDRO(
        data, hidden_dims=[4, 8, 16, 32, 8], seed=SEED, risk='mse'
    )
    gdro.fit(epochs=500, eta=0.001)
    # ---------------------------------------------------------------

    # MaxRM-RF ------------------------------------------------------
    if verbose: print("Fitting MaxRM-RF variants")
    rf_maxrm_mse = modify_rf(copy.deepcopy(rf), "mse", Ytr, Etr, Xte)
    rf_maxrm_nrw = modify_rf(copy.deepcopy(rf), "reward", Ytr, Etr, Xte)
    rf_maxrm_reg = modify_rf(copy.deepcopy(rf), "regret", Ytr, Etr, Xte,
                               fitted_erm, fitted_erm_trees)
    # ---------------------------------------------------------------

    # Collect results ------------------------------------------------
    if verbose: print("Evaluating results")
    all_models = {
        'LR': lr,
        'RF': rf,
        'GroupDRO-NN': gdro,
        'Magging-RF(mse)': rf_magging_mse,
        'Magging-RF(nrw)': rf_magging,
        'Magging-RF(reg)': rf_magging_reg,
        'MaxRM-RF(mse)': rf_maxrm_mse,
        'MaxRM-RF(nrw)': rf_maxrm_nrw,
        'MaxRM-RF(reg)': rf_maxrm_reg,
    }
    results_val = []
    results_test = []
    preds_test = []
    for model_name, model in all_models.items():
        val = True
        for X,Y,E in [(Xval, Yval, Eval), (Xte, Yte, Ete)]:
            if X is None:
                val = False
                continue
            for e in np.unique(E):
                mask = E == e
                Xe = X[mask]
                if model_name == 'GroupDRO-NN':
                    Xe = np.array(Xe)
                preds = model.predict(Xe)
                mse = mean_squared_error(Y[mask], preds)
                out = {
                    "Model": model_name,
                    "EnvIndex": int(e),
                    "MSE": float(mse),
                }
                if val:
                    results_val.append(out)
                else:
                    results_test.append(out)
            val = False

        Xpred = Xte if model_name != 'GroupDRO-NN' else np.array(Xte)
        preds = model.predict(Xpred)
        preds_test.append(pd.DataFrame({
            "Index": np.arange(len(Xte)),
            "Model": model_name,
            "domain_col": Ete,
            "true": Yte,
            "predicted": preds,
            "residual": Yte - preds,
        }))

    if Xval is not None:
        results_val_df = pd.DataFrame(results_val).pivot(
            index="EnvIndex", columns="Model", values="MSE"
        ).reset_index().rename_axis(None, axis=1)
    else:
        results_val_df = None
    results_test_df = pd.DataFrame(results_test).pivot(
        index="EnvIndex", columns="Model", values="MSE"
    ).reset_index().rename_axis(None, axis=1)
    preds_test_df = pd.concat(preds_test, axis=0)
    return results_val_df, results_test_df, preds_test_df


def run_l5go_experiment(X, y, env, unique_envs, seed=42, folds=None, 
                        verbose=True, fold_size=5):
    results_test = []
    preds_test = []

    if folds is None:
        np.random.seed(seed)
        counties_shuffled = np.random.permutation(unique_envs)
        folds = [
            counties_shuffled[i:i+fold_size] 
            for i in range(0, len(unique_envs), fold_size)
        ]

    if len(folds) > 1 and verbose:
        print("--------------------------------")
        print("Median house prices by county per fold:")
        for fold in folds:
            print("\nFold:")
            med_prices = []
            for county in fold:
                mask = env == unique_envs.get_loc(county)
                med_price = np.median(y[mask])
                print(f"\t{county}: {med_price:.2f} (n={np.sum(mask)})")
        
    for fold in range(len(folds)):
        test_counties = folds[fold]
        if verbose:
            print(f"Running experiments with {test_counties} as test environments")
        test_mask = np.isin(env, [unique_envs.get_loc(c) for c in test_counties])
        train_mask = ~test_mask

        X_test, y_test = X[test_mask], y[test_mask]
        X_pool, y_pool = X[train_mask], y[train_mask]
        env_pool = env[train_mask]
        env_test = env[test_mask]

        train_env_indices = np.unique(env_pool)
        X_tr, y_tr, env_tr = X_pool, y_pool, env_pool

        data = DataContainer(n=len(X_tr), N=len(X_test))
        data.load_custom_data(X_tr, y_tr, env_tr, X_test, y_test, env_test)
        _, res_t, preds_t = run_experiment(
            X_tr,
            y_tr,
            env_tr,
            None,
            None,
            None,
            X_test,
            y_test,
            env_test,
            data,
        )
        res_t['HeldOut'] = ", ".join(test_counties)
        preds_t['HeldOut'] = ", ".join(test_counties)
        results_test.append(res_t)
        preds_test.append(preds_t)

    results_test_df = pd.concat(results_test, ignore_index=True)
    preds_test_df = pd.concat(preds_test, ignore_index=True)

    return results_test_df, preds_test_df


#########################################################################
# Main script
#########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity', action='store_true')
    parser.add_argument('--leave-5-out', action='store_true')
    parser.add_argument('--use-geo-clusters', action='store_true')
    parser.add_argument('--many-folds-l5co', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    EXP_SIMILARITY = args.similarity
    EXP_LEAVE_5_COUNTIES_OUT = args.leave_5_out
    USE_GEO_CLUSTERS = args.use_geo_clusters
    EXP_MANY_FOLDS_L5CO = args.many_folds_l5co
    RERUN = args.rerun
    if USE_GEO_CLUSTERS:
        print("Using geographical clusters for experiments.")

    logger.info("Loading data and assigning environments")
    X, y, Z, env, unique_envs = load_data(DATA_DIR, assign_county)

    # ---------------------------------------------------------------------
    # Plot the data points colored by environment
    # ---------------------------------------------------------------------
    plot_env_with_basemap(
        env, Z, 
        out_dir=OUT_DIR,
        label='County',
        counties=unique_envs,
        clustering=[G1, G2, G3, G4, G5] if USE_GEO_CLUSTERS else None
    )

    # ---------------------------------------------------------------------
    # train on i, test on j (to test similarity between envs) 
    # ---------------------------------------------------------------------
    if EXP_SIMILARITY:
        print("Running Experiment: Similarity between environments")
        filepath = os.path.join(OUT_DIR, "env_similarity_results.csv")
        results_df = load_or_compute(
            filepath=filepath, 
            compute_fn=similarity_experiment, 
            args={'unique_envs': unique_envs, 'X': X, 'y': y, 'env': env},
            rerun=RERUN
        ) 
        
        # plot pairwise similarity heatmap
        plot_path = os.path.join(OUT_DIR, "env_similarity_heatmap.png")
        heatmap_data = plot_similarity_results(
            results_df, env, unique_envs, 
            figsize=(6, 5), filepath=plot_path, 
            annot=False,
            clustering=[G1, G2, G3, G4, G5] if USE_GEO_CLUSTERS else None
        )

        # plot the diagonal as a bar plot
        bar_filepath = os.path.join(OUT_DIR, "oob_mse_counties.png")
        plot_oob_mse(heatmap_data, unique_envs, bar_filepath, 
                     clustering=[G1, G2, G3, G4, G5] if USE_GEO_CLUSTERS else None)
 
    # ---------------------------------------------------------------------
    # Leave 5 Counties Out
    # ---------------------------------------------------------------------
    if EXP_LEAVE_5_COUNTIES_OUT:
        print("Running Experiment: Train on 20 counties, test on 5 counties")
        prefix = "geo_" if USE_GEO_CLUSTERS else ""
        if USE_GEO_CLUSTERS:
            OUT_DIR = os.path.join(OUT_DIR, f"split_geo")
            os.makedirs(OUT_DIR, exist_ok=True)

        results_test_df, preds_test_df = load_or_compute(
            filepath=[
                os.path.join(OUT_DIR, prefix+"l5co_results_test.csv"), 
                os.path.join(OUT_DIR, prefix+"l5co_test_preds.csv")
            ],
            compute_fn=run_l5go_experiment,
            args={
                'X': X,
                'y': y,
                'env': env,
                'unique_envs': unique_envs,
                'seed': SEED,
                'folds':[G1, G2, G3, G4, G5] if USE_GEO_CLUSTERS else None
            },
            rerun=RERUN
        )
        for agg in ['mean', 'max']:
            if agg == 'max':
                print_worst_case_environments(
                    results_test_df, unique_envs, models
                )

            generate_tables_and_plots(
                results_test_df, preds_test_df, agg, 
                models, rf_models, colors, OUT_DIR, prefix
            )

        # for each held out, plot a calibration plot of RF and MaxRM-RFs
        calibration_models = ['RF', 'MaxRM-RF(mse)', 'MaxRM-RF(nrw)', 'MaxRM-RF(reg)']
        for held_out in results_test_df['HeldOut'].unique():
            filepath = os.path.join(
                OUT_DIR, 
                f"l5co_calibration_{held_out.replace(', ', '_')}.png"
            )
            calibration_plot(preds_test_df, held_out, calibration_models, colors, filepath)


    # ---------------------------------------------------------------------
    # Many folds of Leave 5 Counties Out
    # ---------------------------------------------------------------------
    if EXP_MANY_FOLDS_L5CO:
        print("Running Experiment: Many folds of Leave 5 Counties Out")
        out_file = os.path.join(
            OUT_DIR, "many_folds_l5co_results.csv")
        preds_file = os.path.join(
            OUT_DIR, "many_folds_l5co_test_preds.csv")
        intermediate_results_dir = os.path.join(
            OUT_DIR, "intermediate_many_folds_l5co")
        os.makedirs(intermediate_results_dir, exist_ok=True)
        
        if not RERUN and os.path.exists(out_file) and os.path.exists(preds_file):
            results_test_df = pd.read_csv(out_file)
            preds_test_df = pd.read_csv(preds_file)
        else:
            results_test = []
            preds_test = []
            for fold_split in range(200):
                split_dir = os.path.join(OUT_DIR, intermediate_results_dir)
                out_file_split = os.path.join(
                    split_dir, f"foldsplit_{fold_split}.csv")
                preds_file_split = os.path.join(
                    split_dir, f"foldsplit_{fold_split}_preds.csv")
                
                np.random.seed(fold_split)
                test_counties = np.random.choice(
                    unique_envs, size=5, replace=False
                )
                
                print(f"Running split {fold_split}")
                res_t, preds_t = load_or_compute(
                    filepath=[out_file_split, preds_file_split],
                    compute_fn=run_l5go_experiment,
                    args={
                        'X': X,
                        'y': y,
                        'env': env,
                        'unique_envs': unique_envs,
                        'seed': SEED,
                        'folds': [test_counties],
                        'verbose': False,
                    },
                    rerun=False
                )
                res_t['fold_split'] = fold_split
                preds_t['fold_split'] = fold_split
                results_test.append(res_t)
                preds_test.append(preds_t)

            results_test_df = pd.concat(results_test, ignore_index=True)
            results_test_df.to_csv(out_file, index=False)
            preds_test_df = pd.concat(preds_test, ignore_index=True)
            preds_test_df.to_csv(preds_file, index=False)            

        print("Aggregated results over many folds of Leave 5 Counties Out:")
        results_test_df_agg = results_test_df.\
            groupby(['HeldOut', 'fold_split']).\
            agg('max').\
            drop(columns=['EnvIndex']).\
            reset_index()
        print_model_comparison_stats(
            results_test_df, 
            ['MaxRM-RF(nrw)', 'MaxRM-RF(reg)', 'MaxRM-RF(mse)']
        )
            
        print("--------------------------------")
        table_df, pval_df = table_test_risk_all_methods_perm(
            results_test_df_agg, preds_test_df, 
            ['RF', 'MaxRM-RF(mse)'],
            folds=True, perm=True, 
            return_p_values=True
            )
        print("% of folds where MaxRM-RF(mse) statistically better than RF:")
        num_better = np.sum(pval_df['MaxRM-RF(mse)'] < 0.05)
        print(f"{num_better} / {len(pval_df)}")
        p_binom = binomtest(
            k=num_better, n=len(pval_df), 
            p=0.5, alternative='greater'
        ).pvalue
        print(f"\tBinomial p-value for MaxRM-RF(mse) better than RF: {p_binom:.4f}")

        print("--------------------------------")
        table_df, pval_df = table_test_risk_all_methods_perm(
            results_test_df_agg, preds_test_df, 
            ['RF', 'MaxRM-RF(mse)'],
            folds=True, perm=True, 
            return_p_values=True,
            alternative='greater'
            )
        print("% of folds where RF statistically better than MaxRM-RF(mse):")
        num_better = np.sum(pval_df['MaxRM-RF(mse)'] < 0.05)
        print(f"{num_better} / {len(pval_df)}")
        p_binom = binomtest(
            k=num_better, n=len(pval_df), 
            p=0.5, alternative='greater'
        ).pvalue
        print(f"\tBinomial p-value for RF better than MaxRM-RF(mse): {p_binom:.4f}")

        # histplot of maximum MSE of RF vs MaxRM-RF(mse)
        maxmse_file = os.path.join(
            OUT_DIR, f"many_folds_l5co_maxrm_vs_rf_max_mse_diff_histplot.png"
        )
        plot_diff_in_max_mse(results_test_df, maxmse_file)
        
  
    # Not used, kept for reference
    # # ---------------------------------------------------------------------
    # # Train on 20, test on 5 (for counties) with bootstrapping
    # # ---------------------------------------------------------------------
    # if EXP_LEAVE_5_COUNTIES_OUT_BOOTSTRAP:
    #     val_file = os.path.join(
    #         OUT_DIR, "l5co_boot_results_val.csv")
    #     test_file = os.path.join(
    #         OUT_DIR, "l5co_boot_results_test.csv")
    #     preds_file = os.path.join(
    #         OUT_DIR, "l5co_boot_test_preds.csv")
        
    #     if RERUN or not os.path.exists(val_file) or not os.path.exists(test_file):
    #         print("Running Experiment: Train on 20 counties, test on 5 counties (with bootstrapping training)")
    #         results_val = []
    #         results_test = []
    #         preds_test = []
    #         np.random.seed(SEED)
    #         counties_shuffled = np.random.permutation(unique_envs)
    #         counties_folds = [counties_shuffled[i:i+5]  for i in range(0, len(unique_envs), 5)]

    #         for i in range(0, len(unique_envs), 5):
    #             test_counties = counties_shuffled[i:i+5]
    #             print(f"Running experiments with {test_counties} as test environments")
    #             test_mask = np.isin(env, [unique_envs.get_loc(c) for c in test_counties])
    #             train_mask = ~test_mask

    #             X_test, y_test = X[test_mask], y[test_mask]
    #             X_pool, y_pool = X[train_mask], y[train_mask]
    #             env_pool = env[train_mask]
    #             env_test = env[test_mask]

    #             train_env_indices = np.unique(env_pool)
    #             for b in tqdm(range(B)):
    #                 X_tr, X_val, y_tr, y_val, env_tr, env_val = train_test_split(
    #                     X_pool,
    #                     y_pool,
    #                     env_pool,
    #                     test_size=VAL_PERCENTAGE,
    #                     random_state=b,
    #                     stratify=env_pool,
    #                 )

    #                 data = DataContainer(n=len(X_tr), N=len(X_test))
    #                 data.load_custom_data(X_tr, y_tr, env_tr, X_test, y_test, env_test)
    #                 res_v, res_t, preds_t = run_experiment(
    #                     X_tr,
    #                     y_tr,
    #                     env_tr,
    #                     X_val,
    #                     y_val,
    #                     env_val,
    #                     X_test,
    #                     y_test,
    #                     env_test,
    #                     data,
    #                 )
    #                 res_v['HeldOut'] = ", ".join(test_counties)
    #                 res_t['HeldOut'] = ", ".join(test_counties)
    #                 preds_t['HeldOut'] = ", ".join(test_counties)
    #                 res_v['B'] = b
    #                 res_t['B'] = b
    #                 preds_t['B'] = b
    #                 results_val.append(res_v)
    #                 results_test.append(res_t)
    #                 preds_test.append(preds_t)

    #         results_val_df = pd.concat(results_val, ignore_index=True)
    #         results_val_df.to_csv(val_file, index=False)
    #         results_test_df = pd.concat(results_test, ignore_index=True)
    #         results_test_df.to_csv(test_file, index=False)
    #         preds_test_df = pd.concat(preds_test, ignore_index=True)
    #         preds_test_df.to_csv(preds_file, index=False)   
    #     else:
    #         results_val_df = pd.read_csv(val_file)
    #         results_test_df = pd.read_csv(test_file)
    #         preds_test_df = pd.read_csv(preds_file)

    #     print("Results test df head:")
    #     print(results_test_df.head())
    #     print("Preds test df head:")
    #     print(preds_test_df.head())
        
    #     for agg in ['mean', 'max']:
    #         results_test_df_agg = results_test_df.\
    #             groupby(['HeldOut', 'B']).\
    #             agg(agg).\
    #             drop(columns=['EnvIndex']).\
    #             reset_index()
    #         print(results_test_df_agg)
    #         table_df = table_test_risk_all_methods(results_test_df_agg, models,
    #                                                folds=DOMAINS=='counties')
    #         latex_str = table_df.to_latex(
    #             index=False, escape=False, column_format="lccccccc"
    #         )
    #         filepath = os.path.join(OUT_DIR, f"l5co_boot_{agg}_all_methods.txt")
    #         print(f"Writing table to {filepath}")
    #         with open(filepath, "w") as f:
    #             f.write(latex_str)

    #         plot_test_risk_all_methods(
    #             results_test_df_agg, models, colors,
    #             saveplot=True, out_dir=OUT_DIR,
    #             nameplot=f"l5co_boot_{agg}_all_methods",
    #             folds=DOMAINS=='counties'
    #         )

    #         plot_test_risk_all_methods(
    #             results_test_df_agg, rf_models, colors,
    #             saveplot=True, out_dir=OUT_DIR,
    #             nameplot=f"l5co_boot_{agg}_rf_methods",
    #             folds=DOMAINS=='counties'
    #         )