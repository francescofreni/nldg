import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from nldg.maximin import MaggingRF
from nldg.utils import gen_data_maximin
from tqdm import tqdm
from utils import plot_mse_r2


def main():
    n_sim = 100
    m_try = 4
    mse = {"RF": [], "MaggingRF": []}
    r2 = {"RF": [], "MaggingRF": []}

    for i in tqdm(range(n_sim)):
        dtr, dts = gen_data_maximin(rng_train=np.random.default_rng(i),
                                    rng_test=np.random.default_rng(i))
        Xtr, Xts = np.array(dtr.drop(columns=['E', 'Y'])), np.array(dts.drop(columns=['E', 'Y']))
        Ytr, Yts = np.array(dtr['Y']), np.array(dts['Y'])
        Etr = np.array(dtr['E'])

        mag_rf = MaggingRF(n_estimators=50, random_state=42, max_features=m_try)
        wpreds = mag_rf.predict_magging(Xtr, Ytr, Etr, Xts)

        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_features=m_try)
        rf.fit(Xtr, Ytr)
        preds = rf.predict(Xts)

        mse['RF'].append(mean_squared_error(Yts, preds))
        mse['MaggingRF'].append(mean_squared_error(Yts, wpreds))
        r2['RF'].append(r2_score(Yts, preds))
        r2['MaggingRF'].append(r2_score(Yts, wpreds))

    # Plot
    mse_df = pd.DataFrame(mse)
    r2_df = pd.DataFrame(r2)
    plot_mse_r2(mse_df, r2_df, "experiments_magging.pdf", name_method='magging')


if __name__ == "__main__":
    main()
