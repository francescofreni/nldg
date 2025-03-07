import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from nldg.old.time.isd import IsdRF
from nldg.old.time.utils import generate_nonlinear_data
from tqdm import tqdm
from experiments.old.utils import plot_mse_r2
from scipy.stats import ortho_group


def main():
    n_sim = 20
    mse = {"RF": [], "IsdRF": []}
    r2 = {"RF": [], "IsdRF": []}

    n_train = 1000
    n_test = 500
    p = 10
    m_train = 10
    m_test = 2
    block_sizes = [2, 4, 3, 1]
    c_coeffs = list(range(2, 9))
    rng = np.random.default_rng(42)
    OM = ortho_group.rvs(dim=p, random_state=rng)
    rng = np.random.default_rng(0)

    for i in tqdm(range(n_sim)):
        Xtr, Ytr, Sigma_list_tr = generate_nonlinear_data(
            n_train, p, m_train, block_sizes, c_coeffs, OM, rng
        )
        Xts, Yts, Sigma_list_ts = generate_nonlinear_data(
            n_test, p, m_test, block_sizes, c_coeffs, OM, rng, test=True
        )

        ws = int(n_train / 8)
        n_rw = 25
        isd_rf = IsdRF(Xtr, Ytr, [ws] * n_rw)
        isd_rf.find_invariant()
        print(isd_rf.th_opt)
        preds_isd_rf = isd_rf.predict_zeroshot(Xts)

        rf = RandomForestRegressor(
            n_estimators=50, random_state=42, max_features=1.0
        )
        rf.fit(Xtr, Ytr)
        preds_rf = rf.predict(Xts)

        mse["RF"].append(mean_squared_error(Yts, preds_rf))
        mse["IsdRF"].append(mean_squared_error(Yts, preds_isd_rf))
        r2["RF"].append(r2_score(Yts, preds_rf))
        r2["IsdRF"].append(r2_score(Yts, preds_isd_rf))

    # Plot
    mse_df = pd.DataFrame(mse)
    r2_df = pd.DataFrame(r2)
    plot_mse_r2(mse_df, r2_df, "experiments_isd_time.pdf", name_method="isd")


if __name__ == "__main__":
    main()
