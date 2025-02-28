import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from nldg.isd import IsdRF
from nldg.utils import gen_data_isd
from tqdm import tqdm
from experiments.utils import plot_mse_r2


def main():
    n_sim = 100
    mse = {"RF": [], "IsdRF": []}
    r2 = {"RF": [], "IsdRF": []}

    for i in tqdm(range(n_sim)):
        dtr, dts, _, _ = gen_data_isd(rng_train=np.random.default_rng(i),
                                      rng_test=np.random.default_rng(i))
        Xtr, Xts = dtr.drop(columns=['E', 'Y']), dts.drop(columns=['E', 'Y'])
        Ytr, Yts = dtr['Y'], dts['Y']

        isd_rf = IsdRF()
        isd_rf.fit_isd(dtr)
        preds_isd_rf = isd_rf.predict_isd(Xts)

        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_features=1.0)
        rf.fit(Xtr, Ytr)
        preds_rf = rf.predict(Xts)

        mse['RF'].append(mean_squared_error(Yts, preds_rf))
        mse['IsdRF'].append(mean_squared_error(Yts, preds_isd_rf))
        r2['RF'].append(r2_score(Yts, preds_rf))
        r2['IsdRF'].append(r2_score(Yts, preds_isd_rf))

    # Plot
    mse_df = pd.DataFrame(mse)
    r2_df = pd.DataFrame(r2)
    plot_mse_r2(mse_df, r2_df, "experiments_isd.pdf")


if __name__ == "__main__":
    main()
