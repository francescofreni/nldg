import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


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
            max_features=max_features,
        )

    def predict_maximin(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        eps: float = 1e-6,
        wtype: str = "inv",
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
            boot_idxs = resample(
                range(n_train),
                replace=True,
                n_samples=n_train,
                random_state=tree.random_state,
            )
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
        if wtype == "inv":
            weights = 1 / (errs + eps)
        else:
            weights = np.exp(-errs)
        weights /= np.sum(weights)
        return weights


class MaggingRF:
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

        Args:
            n_estimators: The number of trees in the forest.
            random_state: Controls the randomness of the bootstrapping of the samples used when building
                trees and the sampling of the features to consider when looking for the best split at each node.
            max_features: The number of features to consider when looking for the best split:
                - If int, then consider max_features features at each split.
                - If “sqrt”, then max_features=sqrt(n_features).
                - If “log2”, then max_features=log2(n_features).
                - If None or 1.0, then max_features=n_features.
                - If float, then max(1, int(max_features * n_features_in_)) features are considered at each split.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.weights_magging = None
        self.n_envs = 0

    def predict_magging(
        self,
        Xtr: np.ndarray,
        Ytr: np.ndarray,
        Etr: np.ndarray,
        Xts: np.ndarray,
    ) -> np.ndarray:
        """
        Fits the Random Forest for each environment and finds the weights for each model.

        Args:
            Xtr: Feature matrix of the training data.
            Ytr: Target values of the training data.
            Etr: Environment labels of the training data.
            Xts: Feature matrix of the test data.

        Returns:
            wpreds: The weighted predictions relative to the test data.
        """
        Xtr_v2, Xval, Ytr_v2, Yval, Etr_v2, Eval = train_test_split(
            Xtr, Ytr, Etr, test_size=0.25, stratify=Etr, random_state=42
        )

        preds_default_val = []
        preds_default_test = []

        unique_envs = np.unique(Etr_v2)
        self.n_envs = len(unique_envs)

        for env in unique_envs:
            rf_env = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )

            env_idx_train = np.where(Etr_v2 == env)[0]

            rf_env.fit(Xtr_v2[env_idx_train], Ytr_v2[env_idx_train])

            preds_default_val.append(rf_env.predict(Xval))
            preds_default_test.append(rf_env.predict(Xts))

        preds_default_val = np.column_stack(preds_default_val)
        preds_default_test = np.column_stack(preds_default_test)

        self.weights_magging = self._find_weights(preds_default_val)
        wpreds = np.dot(self.weights_magging, preds_default_test.T)

        return wpreds

    def _find_weights(
        self,
        preds_default_val: np.ndarray,
    ) -> np.ndarray:
        """
        Finds the weights for each Random Forest regressor using magging.

        Args:
            preds_default_val: Predictions for the validation set for each Random Forest regressor.

        Returns:
            weights_magging: Weights of each Random Forest regressor protecting against the worst-case performance.
        """

        def objective(
            w: np.ndarray,
            F: np.ndarray,
        ) -> float:
            """
            Computes the quadratic form w^T * (F^T F) * w.

            Args:
                w (np.ndarray): Weight vector for the regressors.
                F (np.ndarray): Predictions for the validation set.

            Returns:
                float: The computed objective value.
            """
            return np.dot(w.T, np.dot(F.T, F).dot(w))

        weights_default = np.array([1 / self.n_envs] * self.n_envs)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(preds_default_val.shape[1])]

        weights_magging = minimize(
            objective,
            weights_default,
            args=(preds_default_val,),
            bounds=bounds,
            constraints=constraints,
        ).x

        return weights_magging
