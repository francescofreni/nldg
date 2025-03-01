import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from nldg.maximin import MaximinRF
from nldg.utils import generate_data_example_1, gen_data_maximin
from tqdm import tqdm
from utils import plot_mse_r2


def sim_step(
    dtr: pd.DataFrame,
    dts: pd.DataFrame,
    m_try: int | str | None | float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs one step of the simulation.

    Args:
         dtr: Dataframe containing the training data.
         dts: Dataframe containing the test data.
         m_try: The number of features to consider when looking for the best split:
                - If int, then consider max_features features at each split.
                - If “sqrt”, then max_features=sqrt(n_features).
                - If “log2”, then max_features=log2(n_features).
                - If None or 1.0, then max_features=n_features.
                - If float, then max(1, int(max_features * n_features_in_)) features are considered at each split.

    Returns:
        tuple of 3 arrays:
        - Yts: Response in the test data.
        - preds: Predictions obtained with default Random Forest.
        - wpreds: Weighted predictions obtained with Maximin Random Forest.
    """
    Xtr, Xts = np.array(dtr.drop(columns=['E', 'Y'])), np.array(dts.drop(columns=['E', 'Y']))
    Ytr, Yts = np.array(dtr['Y']), np.array(dts['Y'])
    Etr = np.array(dtr['E'])

    rf = MaximinRF(n_estimators=50, random_state=42, max_features=m_try)
    rf.fit(Xtr, Ytr)

    preds = rf.predict(Xts)
    wpreds, weights = rf.predict_maximin(Xtr, Ytr, Etr, Xts, wtype='inv')

    return Yts, preds, wpreds


def main():
    n_sim = 100

    # Simulation 1
    m_try = 4
    mse_1 = {"RF": [], "MaximinRF": []}
    r2_1 = {"RF": [], "MaximinRF": []}

    for i in tqdm(range(n_sim)):
        dtr, dts = gen_data_maximin(rng_train=np.random.default_rng(i),
                                    rng_test=np.random.default_rng(i))
        Yts, preds, wpreds = sim_step(dtr, dts, m_try)

        mse_1['RF'].append(mean_squared_error(Yts, preds))
        mse_1['MaximinRF'].append(mean_squared_error(Yts, wpreds))
        r2_1['RF'].append(r2_score(Yts, preds))
        r2_1['MaximinRF'].append(r2_score(Yts, wpreds))

    # Plot 1
    mse_df = pd.DataFrame(mse_1)
    r2_df = pd.DataFrame(r2_1)
    plot_mse_r2(mse_df, r2_df, "experiments_maximin_1.pdf")

    # Simulation 2
    mse_2 = {"RF": [], "MaximinRF": []}
    r2_2 = {"RF": [], "MaximinRF": []}

    for i in tqdm(range(n_sim)):
        dtr, dts = generate_data_example_1(i, False)
        Yts, preds, wpreds = sim_step(dtr, dts, m_try)

        mse_2['RF'].append(mean_squared_error(Yts, preds))
        mse_2['MaximinRF'].append(mean_squared_error(Yts, wpreds))
        r2_2['RF'].append(r2_score(Yts, preds))
        r2_2['MaximinRF'].append(r2_score(Yts, wpreds))

    # Plot 2
    mse_df = pd.DataFrame(mse_2)
    r2_df = pd.DataFrame(r2_2)
    plot_mse_r2(mse_df, r2_df, "experiments_maximin_2.pdf")


if __name__ == "__main__":
    main()
