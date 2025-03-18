import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from tqdm import tqdm
from nldg.utils.jbd import ajbd


# =======
# MAGGING
# =======
class MaggingRF(RandomForestRegressor):
    """
    Distribution Generalization with Random Forest Regressor.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        max_features: int | str | None | float = 1.0,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
    ) -> None:
        """
        Initialize the class instance.

        For default parameters, see documentation of `sklearn.ensemble.RandomForestRegressor`.
        """
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )

    def predict_maximin(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions for a given test dataset
        promoting those trees in the Random Forest that
        minimize the OOB prediction error.

        Args:
            X_train: Feature matrix of the training data.
            X_test: Feature matrix of the test data.

        Returns:
            A tuple of 2 numpy arrays:
            - wpreds: Weighted predictions for the test data.
            - weights: Weight for each tree in the forest.
        """
        weights = self._compute_weights(X_train)
        preds = np.array([tree.predict(X_test) for tree in self.estimators_])
        wpreds = np.dot(weights, preds)
        return wpreds, weights

    def _compute_weights(
        self,
        X_train: np.ndarray,
    ) -> np.ndarray:
        """
        Computes weights for each decision tree in the forest based on their
        worst-case mean squared error (MSE) across different environments.

        Args:
            X_train: Feature matrix of the training data.

        Returns:
            weights: Weight for each tree in the forest.
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

        predictions = np.zeros((X_train.shape[0], self.n_estimators))
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X_train)

        weights_dft = np.array([1 / self.n_estimators] * self.n_estimators)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [[0, 1] for _ in range(predictions.shape[1])]

        weights_magging = minimize(
            objective,
            weights_dft,
            args=(predictions,),
            bounds=bounds,
            constraints=constraints,
        ).x

        return weights_magging


# =======
# MAXIMIN
# =======
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
    ) -> float:
        """
        Compute the impurity of a node according to the specified criterion.

        Args:
            y: Response vector.
            E: Environment labels, only used if criterion is "maximin".

        Returns:
            max_err: Impurity of the node.
        """
        if self.criterion == "mse":
            max_err = np.sum((y - np.mean(y)) ** 2)
            return max_err
        else:
            max_mean_err = 0
            max_sum_err = 0
            for env in np.unique(E):
                if np.sum(E == env) > 0:
                    y_e = y[E == env]
                    m_e = np.mean(y_e)
                    sum_err = np.sum((y_e - m_e) ** 2)
                    mean_err = np.mean((y_e - m_e) ** 2)
                    # max_sum_err = max(max_sum_err, sum_err)
                    if mean_err > max_mean_err:
                        max_mean_err = mean_err
                        max_sum_err = sum_err
            return max_sum_err

    def _split_cost(
        self,
        y: np.ndarray,
        E: np.ndarray,
        left_idx: np.ndarray,
        right_idx: np.ndarray,
    ) -> float:
        """
        Compute the cost of a split.

        Args:
            y: Response vector.
            E: Environment labels.
            left_idx: Indices of the observations to the left of the split value.
            right_idx: Indices of the observations to the right of the split value.

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
        left_err = self._node_impurity(left_y, left_E)
        right_err = self._node_impurity(right_y, right_E)
        cost = (left_err + right_err) / len(y)
        # cost = left_err + right_err
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
    ) -> tuple[int, float, float]:
        """
        Find the best split.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.

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
                cost = self._split_cost(y, E, left_idx, right_idx)
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
    ) -> dict:
        """
        Build a tree recursively.

        Args:
            X: Feature matrix.
            y: Response vector.
            E: Environment labels.
        """
        n = len(y)
        if (
            self.max_depth is not None and depth >= self.max_depth
        ) or n < self.min_samples_split:
            return {"pred": np.mean(y)}

        impurity = self._node_impurity(y, E)
        feat, thr, cost = self._best_split(X, y, E)
        if cost >= impurity:
            return {"pred": np.mean(y)}

        left_idx = X[:, feat] <= thr
        right_idx = X[:, feat] > thr

        if (
            np.sum(left_idx) < self.min_samples_leaf
            or np.sum(right_idx) < self.min_samples_leaf
        ):
            return {"pred": np.mean(y)}

        left_tree = self._build_tree(
            X[left_idx], y[left_idx], E[left_idx], depth + 1
        )
        right_tree = self._build_tree(
            X[right_idx], y[right_idx], E[right_idx], depth + 1
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

        self.tree = self._build_tree(X, y, E, 0)

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
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
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


class IsdRF:
    """
    Invariant Subspace Decomposition for Random Forests.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        max_features: int | float | str | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            n_estimators: The number of trees in the forest.
            max_depth: The maximum depth of the tree. If None, then nodes are expanded
                until all leaves are pure or until all leaves contain less than
                min_samples_split observations.
            min_samples_split: The minimum number of observations required to split an internal node.
            min_samples_leaf: The minimum number of observations required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at
                least `min_samples_leaf` training observations in each of the left and
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.Xtr = None
        self.Ytr = None
        self.const_idxs = []
        self.U = None
        self.blocks_shape = None
        self.th_const = None
        self.th_opt = None

    def find_invariant(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
    ) -> None:
        """
        Find the invariant subspace from the training set.

        Args:
            X: The training input observations.
            Y: The target values.
            E: Environment labels.
        """

        def comp_thresholds(X, Y, E, rf):
            n_envs = len(np.unique(E))
            c = np.zeros((n_envs,))
            v = np.zeros((n_envs,))
            for e in range(n_envs):
                X_e = X[E == e]
                Y_e = Y[E == e]
                fitted_values = rf.predict(X_e)
                cov_w = np.cov(
                    Y_e - fitted_values, fitted_values, rowvar=False
                )
                c[e] = cov_w[0, 1]
                v[e] = np.sqrt(cov_w[0, 0] * cov_w[1, 1])

            T = np.mean([np.abs((c[e]) / v[e]) for e in range(n_envs)])
            return T

        def check_const(blocks_shape, th_const, th):
            const_blocks = np.zeros(len(blocks_shape), dtype=bool)
            const_idxs = []
            v_idxs = []
            for b, bs in enumerate(blocks_shape):
                if b == 0:
                    block_idxs = list(range(bs))
                else:
                    block_idxs = [j + sum(blocks_shape[:b]) for j in range(bs)]
                if th_const[b] < th:
                    const_blocks[b] = True
                    for idx in block_idxs:
                        const_idxs.append(idx)
                else:
                    for idx in block_idxs:
                        v_idxs.append(idx)
            return const_blocks, const_idxs, v_idxs

        def xpl_var(Y, Yhat):
            n = len(Y)
            return (1 / n) * (2 * Y.T @ Yhat - Yhat.T @ Yhat)

        self.Xtr = X
        self.Ytr = Y
        n, p = X.shape
        n_envs = len(np.unique(E))

        # Compute covariance matrices
        Sigma = np.zeros((n_envs, p, p))
        for i, e in enumerate(np.unique(E)):
            n_e = np.sum(E == e)
            X_e = X[(i * n_e) : ((i + 1) * n_e)]
            Sigma[i, :, :] = np.cov(X_e, rowvar=False)

        # Joint block diagonalization
        U, blocks_shape, Sigma_diag, _, _ = ajbd(Sigma)
        self.U = U
        self.blocks_shape = blocks_shape

        # Compute constant thresholds for every estimated block
        th_const = []
        for b, bs in enumerate(self.blocks_shape):
            if b == 0:
                block_idxs = list(range(bs))
            else:
                block_idxs = [
                    j + sum(self.blocks_shape[:b]) for j in range(bs)
                ]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            rf.fit(self.Xtr @ (self.U.T[:, block_idxs]), self.Ytr)
            th_const.append(
                comp_thresholds(
                    self.Xtr @ (self.U.T[:, block_idxs]),
                    self.Ytr,
                    E,
                    rf,
                )
            )
        th_const.append(1)
        self.th_const = th_const

        # Invariant parameter estimation
        # Cross-validation for threshold selection
        cv_score = np.zeros(len(th_const))
        cv_score_th = np.zeros((len(th_const), len(np.unique(E))))
        for th_idx, th in enumerate(th_const):
            const_blocks, const_idxs, v_idxs = check_const(
                self.blocks_shape, th_const, th
            )
            self.const_idxs = const_idxs
            for i, e in enumerate(np.unique(E)):
                Xtr_tmp = self.Xtr[E != e]
                Ytr_tmp = self.Ytr[E != e]
                Xts_tmp = self.Xtr[E == e]
                Yts_tmp = self.Ytr[E == e]
                n_ts = Xts_tmp.shape[0]
                n_ad = int(0.2 * n_ts)
                Xad_tmp = Xts_tmp[:n_ad]
                Yad_tmp = Yts_tmp[:n_ad]
                Xts_tmp = Xts_tmp[n_ad:]
                Yts_tmp = Yts_tmp[n_ad:]
                preds_tmp = self.adaption(
                    Xtr_tmp, Ytr_tmp, Xad_tmp, Yad_tmp, Xts_tmp
                )
                cv_score_th[th_idx, i] = xpl_var(Yts_tmp, preds_tmp)
            cv_score[th_idx] = np.mean(cv_score_th[th_idx, :])

        th_opt = th_const[np.argmax(cv_score)]
        self.th_opt = th_opt
        const_blocks, const_idxs, v_idxs = check_const(
            self.blocks_shape, th_const, th_opt
        )
        self.const_idxs = const_idxs

    def predict_zeroshot(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the predictions obtained using the invariant subspace.

        Args:
            X: Test set.

        Returns:
            preds: The predicted values.
        """
        idxs = np.array(self.const_idxs)
        if len(idxs) == 0:
            preds = np.zeros(X.shape[0])
        else:
            X_inv = self.Xtr @ self.U.T[:, idxs]
            rfr = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            rfr.fit(X_inv, self.Ytr)
            preds = rfr.predict((X @ self.U.T[:, idxs]))
        return preds

    def adaption(
        self,
        Xtr: np.ndarray,
        Ytr: np.ndarray,
        Xad: np.ndarray,
        Yad: np.ndarray,
        Xts: np.ndarray,
    ) -> np.ndarray:
        """
        Performs the adaption step.

        Args:
            Xtr: Training data.
            Ytr: Response vector in training data.
            Xad: Adaption data.
            Yad: Response vector in adaption data.
            Xts: Test data.

        Returns:
            preds: The predicted values.
        """
        const_idxs = np.array(self.const_idxs)
        var_idxs = np.array(
            list(set(np.arange(self.Xtr.shape[1])) - set(const_idxs))
        )

        if (
            len(const_idxs) == 0
        ):  # no invariant part, remains only the residual part
            rfr_res = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            rfr_res.fit(Xad @ self.U.T[:, var_idxs], Yad)
            preds = rfr_res.predict(Xts @ self.U.T[:, var_idxs])

        else:
            X_inv = Xtr @ self.U.T[:, const_idxs]
            rfr = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            rfr.fit(X_inv, Ytr)
            if len(const_idxs) == Xtr.shape[1]:
                preds = rfr.predict(Xts @ self.U.T[:, const_idxs])
            else:
                preds_ad_inv = rfr.predict((Xad @ self.U.T[:, const_idxs]))
                res_ad = Yad - preds_ad_inv
                rfr_res = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state,
                )
                rfr_res.fit(Xad @ self.U.T[:, var_idxs], res_ad)
                preds = rfr.predict(
                    Xts @ self.U.T[:, const_idxs]
                ) + rfr_res.predict(Xts @ self.U.T[:, var_idxs])

        return preds
