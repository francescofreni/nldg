from adaXT.random_forest import RandomForest
from nldg.utils import *
from tqdm import tqdm

N_SIM = 50
SAMPLE_SIZE = 900
NOISE_STD = 0.5
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "output_additional")
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":
    data_settings = np.arange(1, 8)
    for data_setting in data_settings:
        results_dict = {
            "RF": [],
            "MaxRM-RF(mse)": [],
        }
        output_path = os.path.join(OUT_DIR, "datasets_comparison.txt")
        for i in tqdm(range(N_SIM)):
            if data_setting == 1:
                dtr = gen_data_v2(n=SAMPLE_SIZE, random_state=i)
                dte = gen_data_v2(n=SAMPLE_SIZE, random_state=1000 + i)
            elif data_setting == 2:
                dtr = gen_data_v3(n=SAMPLE_SIZE, random_state=i, setting=2)
                dte = gen_data_v3(
                    n=SAMPLE_SIZE, random_state=1000 + i, setting=2
                )
            elif data_setting == 3:
                n_easy = SAMPLE_SIZE // 3
                n_hard = SAMPLE_SIZE - n_easy
                dtr = gen_data_v4(n_easy=n_easy, n_hard=n_hard, random_state=i)
                dte = gen_data_v4(
                    n_easy=n_easy, n_hard=n_hard, random_state=1000 + i
                )
            elif data_setting == 4:
                dtr = gen_data_v5(
                    n_samples=SAMPLE_SIZE,
                    adv_fraction=0.1,
                    noise_var_env2=2,
                    setting=2,
                    random_state=i,
                )
                dte = gen_data_v5(
                    n_samples=SAMPLE_SIZE,
                    adv_fraction=0.1,
                    noise_var_env2=2,
                    setting=2,
                    random_state=1000 + i,
                )
            elif data_setting == 5:
                dtr = gen_data_v6(
                    n=SAMPLE_SIZE,
                    noise_std=NOISE_STD,
                    random_state=i,
                    setting=2,
                )
                dte = gen_data_v6(
                    n=SAMPLE_SIZE,
                    noise_std=NOISE_STD,
                    random_state=1000 + i,
                    setting=2,
                )
            elif data_setting == 6:
                dtr = gen_data_v7(n=SAMPLE_SIZE, random_state=i)
                dte = gen_data_v7(n=SAMPLE_SIZE, random_state=1000 + i)
            else:
                dtr = gen_data_v8(n=SAMPLE_SIZE, random_state=i)
                dte = gen_data_v8(n=SAMPLE_SIZE, random_state=1000 + i)

            Xtr = np.array(dtr.drop(columns=["E", "Y"]))
            Ytr = np.array(dtr["Y"])
            Etr = np.array(dtr["E"])

            Xte = np.array(dte.drop(columns=["E", "Y"]))
            Yte = np.array(dte["Y"])
            Ete = np.array(dte["E"])

            rf = RandomForest(
                "Regression",
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                seed=i,
            )
            rf.fit(Xtr, Ytr)
            preds_rf = rf.predict(Xte)

            rf.modify_predictions_trees(Etr, solver="ECOS")
            preds_maxrm = rf.predict(Xte)

            results_dict["RF"].append(max_mse(Yte, preds_rf, Ete))
            results_dict["MaxRM-RF(mse)"].append(
                max_mse(Yte, preds_maxrm, Ete)
            )

        results = pd.DataFrame(results_dict)
        with open(output_path, "a") as f:
            means = results.mean(axis=0)
            n = results.shape[0]
            stderr = results.std(axis=0, ddof=1) / np.sqrt(n)
            ci_lower = means - 1.96 * stderr
            ci_upper = means + 1.96 * stderr
            f.write(f"Dataset {data_setting}\n")
            for col in results.columns:
                f.write(
                    f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
                )
            f.write("\n")
