import os
import copy
import time
from sklearn.metrics import mean_squared_error
from nldg.utils import *
from nldg.ss import MaxRMSmoothSpline, MaggingSmoothSpline
from tqdm import tqdm
from utils import *

N_SIM = 20
SAMPLE_SIZE = 1000
NOISE_STD = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SIM_DIR = os.path.join(RESULTS_DIR, "output_simulation")
os.makedirs(SIM_DIR, exist_ok=True)
OUT_DIR = os.path.join(SIM_DIR, "sim_smoothsplines")
os.makedirs(OUT_DIR, exist_ok=True)

NAME_SS = "MaxRM-SS"


if __name__ == "__main__":
    results_dict = {
        "SS": [],
        "SS(magging)": [],
        f"{NAME_SS}": [],
    }

    runtime_dict = copy.deepcopy(results_dict)
    mse_envs_dict = copy.deepcopy(results_dict)
    maxmse_dict = copy.deepcopy(results_dict)
    mse_dict = copy.deepcopy(results_dict)

    for i in tqdm(range(N_SIM)):
        dtr = gen_data_v6(
            n=SAMPLE_SIZE,
            noise_std=NOISE_STD,
            random_state=i,
            setting=2,
            new_x=True,
        )
        Xtr = np.array(dtr.drop(columns=["E", "Y"]))
        Ytr = np.array(dtr["Y"])
        Etr = np.array(dtr["E"])

        dte = gen_data_v6(
            n=SAMPLE_SIZE,
            noise_std=NOISE_STD,
            random_state=1000 + i,
            setting=2,
            new_x=True,
        )
        Xte = np.array(dte.drop(columns=["E", "Y"]))
        Yte = np.array(dte["Y"])
        Ete = np.array(dte["E"])

        # SS
        start = time.process_time()
        erm_ss = MaxRMSmoothSpline(Xtr, Ytr, cv=True, method="erm")
        end = time.process_time()
        time_ss = end - start
        runtime_dict["SS"].append(time_ss)
        preds_ss = erm_ss.predict(Xte)
        mse_dict["SS"].append(mean_squared_error(Yte, preds_ss))
        mse_envs_ss, maxmse_ss = max_mse(Yte, preds_ss, Ete, ret_ind=True)
        mse_envs_dict["SS"].append(mse_envs_ss)
        maxmse_dict["SS"].append(maxmse_ss)

        # SS - magging
        magging_ss = MaggingSmoothSpline()
        start = time.process_time()
        _ = magging_ss.fit(Xtr, Ytr, Etr)
        end = time.process_time()
        time_ss = end - start
        runtime_dict["SS(magging)"].append(time_ss)
        preds_magging = magging_ss.predict(Xte)
        mse_dict["SS(magging)"].append(mean_squared_error(Yte, preds_magging))
        mse_envs_magging, maxmse_magging = max_mse(
            Yte, preds_magging, Ete, ret_ind=True
        )
        mse_envs_dict["SS(magging)"].append(mse_envs_magging)
        maxmse_dict["SS(magging)"].append(maxmse_magging)

        # MaxRM SS
        start = time.process_time()
        MaxRM_ss = MaxRMSmoothSpline(Xtr, Ytr, Etr, cv=True, solver="ECOS")
        end = time.process_time()
        time_maxrmss = end - start
        runtime_dict[f"{NAME_SS}"].append(time_maxrmss)
        preds_maxrmss = MaxRM_ss.predict(Xte)
        mse_dict[f"{NAME_SS}"].append(mean_squared_error(Yte, preds_maxrmss))
        mse_envs_maxrmss, maxmse_maxrmss = max_mse(
            Yte, preds_maxrmss, Ete, ret_ind=True
        )
        mse_envs_dict[f"{NAME_SS}"].append(mse_envs_maxrmss)
        maxmse_dict[f"{NAME_SS}"].append(maxmse_maxrmss)

    # Results
    mse_df = pd.DataFrame(mse_dict)
    mse_envs_df = get_df(mse_envs_dict)
    maxmse_df = pd.DataFrame(maxmse_dict)
    runtime_df = pd.DataFrame(runtime_dict)

    output_path = os.path.join(OUT_DIR, "summary_all.txt")
    with open(output_path, "w") as f:
        # MSE
        means = mse_df.mean(axis=0)
        n = mse_df.shape[0]
        stderr = mse_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        f.write("MSE\n")
        for col in mse_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
            )
        f.write("\n")

        # Max MSE
        means = maxmse_df.mean(axis=0)
        n = maxmse_df.shape[0]
        stderr = maxmse_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        f.write("Max MSE\n")
        for col in maxmse_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
            )
        f.write("\n")

        # Runtime
        means = runtime_df.mean(axis=0)
        n = runtime_df.shape[0]
        stderr = runtime_df.std(axis=0, ddof=1) / np.sqrt(n)
        ci_lower = means - 1.96 * stderr
        ci_upper = means + 1.96 * stderr
        f.write("Runtime (seconds)\n")
        for col in runtime_df.columns:
            f.write(
                f"{col}: mean = {means[col]:.4f}, 95% CI = [{ci_lower[col]:.4f}, {ci_upper[col]:.4f}]\n"
            )

    print(f"Saved results to {OUT_DIR}")
