import numpy as np
from sklearn.ensemble import RandomForestRegressor
from adaXT.random_forest import RandomForest
from scipy.optimize import minimize
from tqdm import tqdm
from joblib import Parallel, delayed


# =======
# MAGGING
# =======
class MaggingRF_PB:
    """
    Distribution Generalization with Random Forest Regressor.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        max_depth: int | None = None,
        max_features: int | str | None | float = 1.0,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        backend: str = "RF4DG",
        disable: bool = False,
        parallel: bool = False,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            For default parameters, see documentation of `sklearn.ensemble.RandomForestRegressor`.
            backend: Backend to use for fitting the forests.
                Possible values are {"RF4DG", "sklearn", "adaXT"}
            disable: Whether to disable the progress bar (useful for running simulations).
                Only available if backend is "RF4DG".
            parallel: Whether to run the algorithm in parallel.
                It can be used only if backend is "RF4DG".
        """
        if backend not in ["RF4DG", "sklearn", "adaXT"]:
            raise ValueError(
                "backend must be one of 'RF4DG', 'sklearn', 'adaXT'."
            )
        if backend != "RF4DG" and parallel:
            raise ValueError(
                "parallel may be set to True when backend is 'RF4DG'."
            )
        if backend != "RF4DG" and disable:
            raise ValueError(
                "disable may be set to True when backend is 'RF4DG'."
            )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.weights_magging = None
        self.backend = backend
        self.disable = disable
        self.parallel = parallel

    def get_weights(self) -> np.ndarray | None:
        return self.weights_magging

    def fit_predict_magging(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions for a given test dataset
        promoting those trees in the Random Forest that
        minimize the OOB prediction error.

        Args:
            X_train: Feature matrix of the training data.
            Y_train: Response vector of the training data.
            E_train: Environment label of the training data.
            X_test: Feature matrix of the test data.

        Returns:
            A tuple of 2 numpy arrays:
            - wfitted: Weighted fitted values.
            - wpreds: Weighted predictions.
        """

        def objective(w: np.ndarray, F: np.ndarray) -> float:
            return np.dot(w.T, np.dot(F.T, F).dot(w))

        n_envs = len(np.unique(E_train))
        winit = np.array([1 / n_envs] * n_envs)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(n_envs)]

        preds_envs = []
        fitted_envs = []
        for env in np.unique(E_train):
            Xtr_e = X_train[E_train == env]
            Ytr_e = Y_train[E_train == env]
            Etr_e = E_train[E_train == env]
            if self.backend == "RF4DG":
                rfm = RF4DG(
                    criterion="mse",
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state,
                    disable=self.disable,
                    parallel=self.parallel,
                )
                rfm.fit(Xtr_e, Ytr_e, Etr_e)
            elif self.backend == "sklearn":
                rfm = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state,
                )
                rfm.fit(Xtr_e, Ytr_e)
            else:
                if self.max_depth is None:
                    self.max_depth = 2**31 - 1
                rfm = RandomForest(
                    forest_type="Regression",
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    seed=self.random_state,
                )
                rfm.fit(Xtr_e, Ytr_e)
            preds_envs.append(rfm.predict(X_test))
            fitted_envs.append(rfm.predict(X_train))
        preds_envs = np.column_stack(preds_envs)
        fitted_envs = np.column_stack(fitted_envs)

        wmag = minimize(
            objective,
            winit,
            args=(fitted_envs,),
            bounds=bounds,
            constraints=constraints,
        ).x
        self.weights_magging = wmag
        wpreds = np.dot(wmag, preds_envs.T)
        wfitted = np.dot(wmag, fitted_envs.T)

        return wfitted, wpreds


# =======
# MAXIMIN
# =======
class DT4DG:
    """
    Decision Tree for Distribution Generalization
    """

    def __init__(
        self,
        criterion: str = "mse",
        max_depth: int | None = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            criterion: {"mse", "maximin"}
                The function to measure the quality of a split. Supported criteria
                are "mse" for the mean squared error, which is equal to variance
                reduction as feature selection criterion and minimizes the L2 loss
                using the mean of each terminal node; "maximin" for the worst case
                mean squared error across environments.
            max_depth: The maximum depth of the tree. If None, then nodes are expanded
                until all leaves are pure or until all leaves contain less than
                min_samples_split observations.
            min_samples_split: The minimum number of observations required to split an internal node.
            min_samples_leaf: The minimum number of observations required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at
                least ``min_samples_leaf`` training observations in each of the left and
                right branches.
            max_features : int, float or {"sqrt", "log2"}, default=None
                The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `max(1, int(max_features * n_features_in_))` features are considered at each
                  split.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
            random_state: Random seed.
        """
        if criterion not in ["mse", "maximin"]:
            raise ValueError("Criterion should be either 'mse' or 'maximin'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def _node_impurity(
        self,
        y: np.ndarray,
        E: np.ndarray,
        e_worst_prev: float | None = None,
    ) -> float | tuple[float, int]:
        """
        Compute the impurity of a node according to the specified criterion.

        Args:
            y: Response vector.
            E: Environment labels, only used if criterion is "maximin".
            e_worst_prev: id of the environment that was worst-case in the previous split.

        Returns:
            max_err: Impurity of the node.
            if self.criterion == "maximin":
                e_worst: id of the environment that leads to the maximum impurity.
        """
        if self.criterion == "mse":
            max_err = np.sum((y - np.mean(y)) ** 2)
            return max_err
        else:
            max_mean_err = 0
            max_sum_err = 0
            e_worst = None
            if e_worst_prev is not None and np.sum(E == e_worst_prev) > 0:
                pred = np.mean(y[E == e_worst_prev])
            else:
                # if e_worst_prev is None, we are in the root node
                # and the prediction should be the overall mean.
                pred = np.mean(y)
            for env in np.unique(E):
                if np.sum(E == env) > 0:
                    y_e = y[E == env]
                    sum_err = np.sum((y_e - pred) ** 2)
                    mean_err = np.mean((y_e - pred) ** 2)
                    if mean_err > max_mean_err:
                        max_mean_err = mean_err
                        max_sum_err = sum_err
                        e_worst = env
            return max_sum_err, e_worst

    def _split_cost(
        self,
        y: np.ndarray,
        E: np.ndarray,
        left_idx: np.ndarray,
        right_idx: np.ndarray,
        e_worst_prev: float | None = None,
    ) -> float:
        """
        Compute the cost of a split.

        Args:
            y: Response vector.
            E: Environment labels.
            left_idx: Indices of the observations to the left of the split value.
            right_idx: Indices of the observations to the right of the split value.
            e_worst_prev: id of the environment that was worst-case in the previous split.

        Returns:
            cost: Cost due to the split.
        """
        if (
            np.sum(left_idx) < self.min_samples_leaf
            or np.sum(right_idx) < self.min_samples_leaf
        ):
            return np.inf
        left_y = y[left_idx]
        right_y = y[right_idx]
        left_E = E[left_idx]
        right_E = E[right_idx]
        if self.criterion == "mse":
            left_err = self._node_impurity(left_y, left_E)
            right_err = self._node_impurity(right_y, right_E)
            cost = (left_err + right_err) / len(y)
        else:
            left_err, e_worst_left = self._node_impurity(
                left_y, left_E, e_worst_prev
            )
            right_err, e_worst_right = self._node_impurity(
                right_y, right_E, e_worst_prev
            )
            n_e_worst_left = np.sum(left_E == e_worst_left)
            n_e_worst_right = np.sum(right_E == e_worst_right)
            cost = (left_err + right_err) / (n_e_worst_left + n_e_worst_right)
        return cost

    def _select_features(
        self,
        n_features: int,
    ) -> int:
        """
        Compute the number of features given the value of max_features.
        """
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            return max(1, min(n_features, int(self.max_features * n_features)))
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError("Invalid string for max_features")
        raise ValueError("max_features must be int, float, str, or None")

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        E: np.ndarray,
        e_worst_prev: float | None = None,
    ) -> tuple[int, float, float]:
        """
        Find the best split.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.
            e_worst_prev: id of the environment that was worst-case in the previous split.

        Returns:
            Tuple of 3 elements:
            - best_feature: The feature index that leads to the best split.
            - best_threshold: The threshold of the best feature that leads to the best split.
            - best_cost: The cost due to the best split.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        best_cost = np.inf
        best_feature = None
        best_threshold = np.inf
        m_try = self._select_features(p)
        feature_idxs = np.random.default_rng(self.random_state).choice(
            p, m_try, replace=False
        )

        for feat in feature_idxs:
            sort_idx = np.argsort(X[:, feat])
            X_sorted = X[sort_idx, feat]
            for i in range(1, n):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2
                left_idx = X[:, feat] <= threshold
                right_idx = X[:, feat] > threshold
                cost = self._split_cost(
                    y, E, left_idx, right_idx, e_worst_prev
                )
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feat
                    best_threshold = threshold
        return best_feature, best_threshold, best_cost

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        E: np.ndarray,
        depth: int,
        e_worst_prev: float | None = None,
    ) -> dict:
        """
        Build a tree recursively.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.
            e_worst_prev: id of the environment that was worst-case in the previous split.
        """
        n = len(y)
        if (
            self.max_depth is not None and depth >= self.max_depth
        ) or n < self.min_samples_split:
            if self.criterion == "mse":
                pred = np.mean(y)
            else:
                if e_worst_prev is not None and np.sum(E == e_worst_prev) > 0:
                    pred = np.mean(y[E == e_worst_prev])
                else:
                    pred = np.mean(y)
            return {"pred": pred}

        if self.criterion == "mse":
            impurity = self._node_impurity(y, E)
            feat, thr, cost = self._best_split(X, y, E)

            if feat is None:
                is_leaf = True
            else:
                left_idx = X[:, feat] <= thr
                right_idx = X[:, feat] > thr
                left_err = self._node_impurity(y[left_idx], E[left_idx])
                right_err = self._node_impurity(y[right_idx], E[right_idx])
                is_leaf = (
                    (impurity <= left_err + right_err)
                    or (np.sum(left_idx) < self.min_samples_leaf)
                    or (np.sum(right_idx) < self.min_samples_leaf)
                )

            if is_leaf:
                return {"pred": np.mean(y)}
        else:
            impurity, e_worst = self._node_impurity(y, E, e_worst_prev)
            feat, thr, cost = self._best_split(X, y, E, e_worst)

            if feat is None:
                is_leaf = True
            else:
                left_idx = X[:, feat] <= thr
                right_idx = X[:, feat] > thr

                left_err, e_worst_left = self._node_impurity(
                    y[left_idx], E[left_idx], e_worst
                )
                right_err, e_worst_right = self._node_impurity(
                    y[right_idx], E[right_idx], e_worst
                )

                is_leaf = (
                    (impurity <= left_err + right_err)
                    or (np.sum(left_idx) < self.min_samples_leaf)
                    or (np.sum(right_idx) < self.min_samples_leaf)
                )

            if is_leaf:
                if np.sum(E == e_worst_prev) > 0:
                    pred = np.mean(y[E == e_worst_prev])
                else:
                    pred = np.mean(y)
                return {"pred": pred}

        if self.criterion == "mse":
            left_tree = self._build_tree(
                X[left_idx], y[left_idx], E[left_idx], depth + 1
            )
            right_tree = self._build_tree(
                X[right_idx], y[right_idx], E[right_idx], depth + 1
            )
        else:
            left_tree = self._build_tree(
                X[left_idx], y[left_idx], E[left_idx], depth + 1, e_worst
            )
            right_tree = self._build_tree(
                X[right_idx], y[right_idx], E[right_idx], depth + 1, e_worst
            )

        return {
            "feat": feat,
            "thr": thr,
            "left": left_tree,
            "right": right_tree,
        }

    def _predict_x(
        self,
        x: np.ndarray,
        tree: dict,
    ) -> float:
        """
        Predict y given an observation. The function traverses the tree until
        a leaf node is found.

        Args:
             x: Observation contained in the feature matrix.
             tree: Node either containing only the prediction if the node is a leaf node
                or the best feature, the best threshold and the nodes to the left and
                right otherwise.
        """
        if "feat" not in tree:
            return tree["pred"]
        if x[tree["feat"]] <= tree["thr"]:
            return self._predict_x(x, tree["left"])
        else:
            return self._predict_x(x, tree["right"])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        E: np.ndarray | None = None,
    ) -> None:
        """
        Build a decision tree regressor from the training set.

        Args:
            X: The training input samples.
            y: The target values.
            E: Environment labels.
        """
        if self.criterion == "maximin":
            if E is None:
                raise ValueError("E is necessary when criterion is 'maximin'.")
        else:
            E = np.zeros(len(y))

        self.tree = self._build_tree(X, y, E, 0, None)

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the regression values for X.

        Args:
            X: The input observations.

        Returns:
            preds: The predicted values.
        """
        preds = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            preds[i] = self._predict_x(x, self.tree)
        return preds


class RF4DG:
    """
    Random Forest for Distribution Generalization.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: int | None = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
        disable: bool = False,
        parallel: bool = False,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            n_estimators: The number of trees in the forest.
            criterion: {"mse", "maximin"}
                The function to measure the quality of a split. Supported criteria
                are "mse" for the mean squared error, which is equal to variance
                reduction as feature selection criterion and minimizes the L2 loss
                using the mean of each terminal node; "maximin" for the worst case
                mean squared error across environments.
            max_depth: The maximum depth of the tree. If None, then nodes are expanded
                until all leaves are pure or until all leaves contain less than
                min_samples_split observations.
            min_samples_split: The minimum number of observations required to split an internal node.
            min_samples_leaf: The minimum number of observations required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at
                least ``min_samples_leaf`` training observations in each of the left and
                right branches.
            max_features : int, float or {"sqrt", "log2"}, default=None
                The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `max(1, int(max_features * n_features_in_))` features are considered at each
                  split.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
            random_state: Random seed.
            disable: Whether to disable the progress bar (useful for running simulations).
            parallel: Whether to run the algorithm in parallel.
        """
        if criterion not in ["mse", "maximin"]:
            raise ValueError("Criterion should be either 'mse' or 'maximin'.")
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.disable = disable
        self.parallel = parallel
        self.forest = [None] * self.n_estimators

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        E: np.ndarray,
    ) -> None:
        """
        Build a forest of trees from the training set (X, y)

        Args:
            X: The training input observations.
            y: The target values.
            E: Environment labels.
        """

        def _fit_tree(
            X: np.ndarray,
            y: np.ndarray,
            E: np.ndarray,
            bootstrap_idx: np.ndarray,
        ) -> DT4DG:
            """
            Fit a Regression tree on a bootstrap sample.
            """
            tree = DT4DG(
                self.criterion,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.max_features,
                self.random_state,
            )
            tree.fit(X[bootstrap_idx], y[bootstrap_idx], E[bootstrap_idx])
            return tree

        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.parallel:
            bootstrap_indices = []
            for _ in range(self.n_estimators):
                idx = rng.choice(n, n, replace=True)
                bootstrap_indices.append(idx)
            self.forest = Parallel(n_jobs=-1)(
                delayed(_fit_tree)(X, y, E, bootstrap_indices[i])
                for i in tqdm(range(self.n_estimators), disable=self.disable)
            )
        else:
            for i in tqdm(range(self.n_estimators), disable=self.disable):
                bootstrap_idx = rng.choice(n, n, replace=True)
                tree = DT4DG(
                    self.criterion,
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    self.max_features,
                    self.random_state,
                )
                tree.fit(X[bootstrap_idx], y[bootstrap_idx], E[bootstrap_idx])
                self.forest[i] = tree

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the regression values for X.

        Args:
            X: The input observations.

        Returns:
            preds: The predicted values.
        """
        preds = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.forest):
            preds[:, i] = tree.predict(X)
        preds = np.mean(preds, axis=1)
        return preds
