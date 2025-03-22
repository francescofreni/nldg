import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


class DT4DL:
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
        random_state: int = 42,
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
        e_worst_pred: float | None = None,
    ) -> float | tuple[float, int]:
        """
        Compute the impurity of a node according to the specified criterion.

        Args:
            y: Response vector.
            E: Environment labels, only used if criterion is "maximin".
            e_worst_pred: id of the environment that was worst-case in the previous split.

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
            if e_worst_pred is not None and np.sum(E == e_worst_pred) > 0:
                pred = np.mean(y[E == e_worst_pred])
            else:
                # if e_worst_pred is None, we are in the root node
                # and the prediction should be the overall mean.
                pred = np.mean(y)
            for env in np.unique(E):
                if np.sum(E == env) > 0:
                    y_e = y[E == env]
                    sum_err = np.sum((y_e - pred) ** 2)
                    mean_err = np.mean((y_e - pred) ** 2)
                    # max_sum_err = max(max_sum_err, sum_err)
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
        e_worst_pred: float | None = None,
    ) -> float:
        """
        Compute the cost of a split.

        Args:
            y: Response vector.
            E: Environment labels.
            left_idx: Indices of the observations to the left of the split value.
            right_idx: Indices of the observations to the right of the split value.
            e_worst_pred: id of the environment that was worst-case in the previous split.

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
        else:
            left_err, e_worst_left = self._node_impurity(
                left_y, left_E, e_worst_pred
            )
            right_err, e_worst_right = self._node_impurity(
                right_y, right_E, e_worst_pred
            )
        cost = (left_err + right_err) / len(
            y
        )  # TODO: You may want to change how you combine the costs
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
        e_worst_pred: float | None = None,
    ) -> tuple[int, float, float]:
        """
        Find the best split.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.
            e_worst_pred: id of the environment that was worst-case in the previous split.

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
        best_threshold = None
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
                    y, E, left_idx, right_idx, e_worst_pred
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
        e_worst_pred: float | None = None,
    ) -> dict:
        """
        Build a tree recursively.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.
            e_worst_pred: id of the environment that was worst-case in the previous split.
        """
        n = len(y)
        if (
            self.max_depth is not None and depth >= self.max_depth
        ) or n < self.min_samples_split:
            if self.criterion == "mse":
                pred = np.mean(y)
            else:
                if e_worst_pred is not None and np.sum(E == e_worst_pred) > 0:
                    pred = np.mean(y[E == e_worst_pred])
                else:
                    pred = np.mean(y)
            return {"pred": pred}

        if self.criterion == "mse":
            impurity = self._node_impurity(y, E)
            feat, thr, cost = self._best_split(X, y, E)
            if cost >= impurity:
                return {"pred": np.mean(y)}
        else:
            impurity, e_worst = self._node_impurity(y, E, e_worst_pred)
            feat, thr, cost = self._best_split(X, y, E, e_worst)
            if cost >= impurity:
                return {"pred": np.mean(y[E == e_worst])}

        left_idx = X[:, feat] <= thr
        right_idx = X[:, feat] > thr

        if (
            np.sum(left_idx) < self.min_samples_leaf
            or np.sum(right_idx) < self.min_samples_leaf
        ):
            if self.criterion == "mse":
                pred = np.mean(y)
            else:
                pred = np.mean(y[E == e_worst])
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


class RF4DL:
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
        random_state: int = 42,
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
        ) -> DT4DL:
            """
            Fit a Regression tree on a bootstrap sample.
            """
            tree = DT4DL(
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
                tree = DT4DL(
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
