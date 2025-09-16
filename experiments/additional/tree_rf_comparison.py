from adaXT.random_forest import RandomForest
from nldg.utils import *

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


def f_opt(x):
    return np.where(x <= 0, 1.25 * x, 2.25 * x)


if __name__ == "__main__":
    dte = gen_data_v6(
        n=SAMPLE_SIZE,
        noise_std=NOISE_STD,
        random_state=1000,
        setting=2,
    )
    Xte = np.array(dte.drop(columns=["E", "Y"]))
    Yte = np.array(dte["Y"])
    Ete = np.array(dte["E"])
    preds_opt = f_opt(Xte).ravel()

    rf_preds = np.empty((N_SIM, SAMPLE_SIZE))
    tree_preds = np.empty((N_SIM, SAMPLE_SIZE))

    for i in range(N_SIM):
        dtr = gen_data_v6(
            n=SAMPLE_SIZE, noise_std=NOISE_STD, random_state=i, setting=2
        )
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        rf = RandomForest(
            "Regression",
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            seed=42,
        )
        rf.fit(Xtr, Ytr)
        rf.modify_predictions_trees(Etr)

        rf_preds[i, :] = rf.predict(Xte)
        tree_preds[i, :] = rf.trees[0].predict(Xte)

    rf_mean = rf_preds.mean(axis=0)
    tree_mean = tree_preds.mean(axis=0)

    rf_var_pt = rf_preds.var(axis=0, ddof=0)
    tree_var_pt = tree_preds.var(axis=0, ddof=0)

    rf_bias2_pt = (rf_mean - preds_opt) ** 2
    tree_bias2_pt = (tree_mean - preds_opt) ** 2

    rf_bias2 = rf_bias2_pt.mean()
    rf_var = rf_var_pt.mean()
    tree_bias2 = tree_bias2_pt.mean()
    tree_var = tree_var_pt.mean()

    output_path = os.path.join(OUT_DIR, "summary_all.txt")
    with open(output_path, "w") as f:
        f.write("MaxRM Random Forest\n")
        f.write(f"Bias^2: {rf_bias2}\n")
        f.write(f"Variance: {rf_var}\n")
        f.write("\nMaxRM regression tree\n")
        f.write(f"Bias^2: {tree_bias2}\n")
        f.write(f"Variance: {tree_var}\n")

    print(f"Saved results to {OUT_DIR}")
