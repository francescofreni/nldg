import time
from nldg.utils import *
from adaXT.random_forest import RandomForest


if __name__ == "__main__":
    dtr = gen_data_v6(n=1000, noise_std=0.5)
    Xtr = np.array(dtr.drop(columns=["E", "Y"]))
    Ytr = np.array(dtr["Y"])
    Etr = np.array(dtr["E"])
    Xtr_sorted = np.sort(Xtr, axis=0)
    n_estimators = 50
    min_samples_leaf = 30
    random_state = 42

    rf = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=random_state,
    )
    rf.fit(Xtr, Ytr)

    start = time.time()
    rf.modify_predictions_trees(Etr)
    end = time.time()
    print("CP, serial:", end - start)

    rf = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=random_state,
    )
    rf.fit(Xtr, Ytr)

    start = time.time()
    rf.modify_predictions_trees_parallel(Etr, n_jobs=50)
    end = time.time()
    print("CP, parallel:", end - start)

    rf = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=random_state,
    )
    rf.fit(Xtr, Ytr)

    start = time.time()
    rf.modify_predictions_trees(Etr, opt_method="extragradient")
    end = time.time()
    print("Extragradient, serial:", end - start)

    rf = RandomForest(
        "Regression",
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        seed=random_state,
    )
    rf.fit(Xtr, Ytr)

    start = time.time()
    rf.modify_predictions_trees_parallel(
        Etr, opt_method="extragradient", n_jobs=50
    )
    end = time.time()
    print("Extragradient, parallel:", end - start)
