import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample


class MaximinRF(RandomForestRegressor):
    """
    Distribution Generalization with Random Forest Regressor.
    """
    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = 42,
        max_features: int | float = 1.0,
    ) -> None:
        """
        Initialize the class instance.

        For default parameters, see documentation of `sklearn.ensemble.RandomForestRegressor`.
        """
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            max_features=max_features
        )

    def predict_maximin(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        eps: float = 1e-6,
        wtype: str = 'inv',
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions for a given test dataset
        promoting those trees in the Random Forest that
        minimize the OOB prediction error.

        Args:
            X_train: Feature matrix of the training data.
            Y_train: Target values of the training data.
            E_train: Environment labels of the training data.
            X_test: Feature matrix of the test data.
            eps: Constant for numerical stability in weight computation.
            wtype: Method to compute the weight. Accepted values are 'inv' and 'soft'.

        Returns:
            A tuple of 2 numpy arrays:
            - wpreds: Weighted predictions for the test data.
            - weights: Weight for each tree in the forest.
        """
        weights = self._compute_weights(X_train, Y_train, E_train, eps, wtype)
        preds = np.array([tree.predict(X_test) for tree in self.estimators_])
        wpreds = np.dot(weights, preds)
        return wpreds, weights

    def _compute_weights(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
        eps: float,
        wtype: str,
    ) -> np.ndarray:
        """
        Computes weights for each decision tree in the forest based on their
        worst-case mean squared error (MSE) across different environments.

        Args:
            X_train: Feature matrix of the training data.
            Y_train: Target values of the training data.
            E_train: Environment labels of the training data.
            eps: Constant for numerical stability in weight computation.
            wtype: Method to compute the weight. Accepted values are 'inv' and 'soft'.

        Returns:
            weights: Weight for each tree in the forest.
        """
        n_train = X_train.shape[0]
        errs = np.zeros(self.n_estimators)
        for i, tree in enumerate(self.estimators_):
            boot_idxs = resample(range(n_train), replace=True, n_samples=n_train, random_state=tree.random_state)
            oob_idxs = np.array(list(set(range(n_train)) - set(boot_idxs)))
            envs = np.unique(E_train[oob_idxs])
            max_err = 0
            for env in envs:
                env_idxs = oob_idxs[E_train[oob_idxs] == env]
                Y_true = Y_train[env_idxs]
                Y_pred = tree.predict(X_train[env_idxs])
                env_error = np.mean((Y_true - Y_pred) ** 2)
                max_err = max(env_error, max_err)
            errs[i] = max_err if len(envs) > 0 else np.inf
        eps = 1e-6
        if wtype == 'inv':
            weights = 1 / (errs + eps)
        else:
            weights = np.exp(-errs)
        weights /= np.sum(weights)
        return weights
