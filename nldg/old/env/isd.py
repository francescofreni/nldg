"""
Code inspired from https://github.com/mlazzaretto/Invariant-Subspace-Decomposition.git
"""

from __future__ import division
import numpy as np
import pandas as pd
from nldg.utils.jbd import jbd, ajbd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold


class IsdRF:
    """
    Invariant Subspace Decomposition for Random Forests.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = 42,
        max_features: int | str | None | float = 1.0,
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
        self.data_train = None
        self.X_train = None
        self.Y_train = None
        self.E_train = None
        self.n_train = 0
        self.n_envs_train = 0
        self.p = 0

        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features

        self.c_blocks = None
        self.c_idxs = list([self.p])
        self.v_idxs = list([self.p])
        self.U = np.zeros((self.p, self.p))
        self.Sigma_diag = None
        self.blocks_shape = []
        self.th_const = []
        self.th_opt = 0
        self.var_th = None
        self.var = None

    # TODO: In the following, to find the invariant subspace, I first project the features on
    #  the candidate space and then fit the Random Forest. Should I first fit the Random Forest and
    #  then project the fitted values instead?

    def _comp_thresholds(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
    ) -> float:
        """
        Compute invariance threshold T for constant coefficient selection for a given block.

        Args:
            X: observed predictors.
            Y: response.
            E: environment labels.

        Returns:
            T: invariance threshold.
        """
        n_envs = len(np.unique(E))
        c = np.zeros((n_envs,))
        v = np.zeros((n_envs,))
        for i in range(n_envs):
            X_e = X[E == i, :]
            Y_e = Y[E == i]
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(X_e, Y_e)
            fitted_values = rf.predict(X_e)
            cov_e = np.cov(Y_e - fitted_values, fitted_values, rowvar=False)
            c[i] = cov_e[0, 1]
            v[i] = np.sqrt(cov_e[0, 0] * cov_e[1, 1])

        T = np.mean([np.abs((c[i]) / v[i]) for i in range(n_envs)])
        return T

    def _check_const(
        self,
        blocks_shape: list,
        th_const: list,
        th: float,
    ) -> tuple[np.array, list, list]:
        """
        Check which blocks are invariant for a given threshold.

        Args:
            blocks_shape: List of blocks dimensions.
            th_const: List of thresholds corresponding to each block.
            th: Reference threshold to check.

        Returns:
            A tuple of 3 elements:
            - const_blocks: Boolean vector for constant blocks.
            - const_idxs: List of covariates indices corresponding to constant blocks.
            - v_idxs: List of covariates indices corresponding to environment-varying blocks
        """
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
        self.c_blocks = const_blocks
        self.c_idxs = const_idxs
        self.v_idxs = v_idxs
        return const_blocks, const_idxs, v_idxs

    def fit_isd(
        self,
        data_train: pd.DataFrame,
        k_fold: int | None = None,
        diag: bool = False,
    ) -> None:
        """
        Fit a Random Forest after performing Invariant Subspace Decomposition.

        Args:
            data_train: Train data containing the features, the response and the environment label.
            k_fold: Number of folds for cross-validation for threshold selection.
            diag: True if the estimated covariance matrices are assumed to be jointly diagonalizable.
        """
        X_train_df = data_train.drop(columns=["E", "Y"])
        Y_train_df = data_train["Y"]
        E_train_df = data_train["E"]

        self.data_train = data_train
        self.X_train = np.array(X_train_df)
        self.Y_train = np.ravel(Y_train_df)
        self.E_train = np.array(E_train_df).flatten()
        self.n_train = data_train.shape[0]
        self.n_envs_train = len(np.unique(E_train_df))
        self.p = self.X_train.shape[1]

        # 1) Compute the covariance matrices
        Sigma = np.stack(
            self.data_train.drop(columns=["E", "Y"])
            .groupby(self.data_train["E"])
            .apply(lambda g: g.cov().values)
        )

        # TODO: if I were to fit the Random Forest first and then project the fitted values,
        #  I would have to do it at this point.

        # 2) Joint Block Diagonalization
        if diag:
            U, _, _, _ = jbd(Sigma, threshold=0, diag=True)
            blocks_shape = list[np.ones(self.p)]
            Sigma_diag = np.zeros_like(Sigma)
            for k in range(self.n_envs_train):
                Sigma_diag[k, :, :] = U @ Sigma[k, :, :] @ U.T.conj()
        else:
            U, blocks_shape, Sigma_diag, _, _ = ajbd(Sigma)

        self.U = U
        self.Sigma_diag = Sigma_diag

        # Compute constant thresholds for every estimated block
        th_const = []
        for b, bs in enumerate(blocks_shape):
            if b == 0:
                block_idxs = list(range(bs))
            else:
                block_idxs = [j + sum(blocks_shape[:b]) for j in range(bs)]

            th_const.append(
                self._comp_thresholds(
                    self.X_train @ (U[:, block_idxs] @ U[:, block_idxs].T),
                    self.Y_train,
                    self.E_train,
                )
            )
        th_const.append(1)

        # 3) Search for invariant subspace
        # Cross-validation for threshold selection
        if not k_fold:
            k_fold = len(np.unique(self.E_train))

        var = np.zeros(len(th_const))
        var_th = np.zeros((len(th_const), k_fold))

        for th_idx, th in enumerate(th_const):
            const_blocks, const_idxs, v_idxs = self._check_const(
                blocks_shape, th_const, th
            )
            skf = StratifiedKFold(
                n_splits=k_fold, shuffle=True, random_state=42
            )

            for fd_idx, (train_idx, val_idx) in enumerate(
                skf.split(self.X_train, self.E_train)
            ):
                X_train_fold, X_val_fold = (
                    self.X_train[train_idx],
                    self.X_train[val_idx],
                )
                Y_train_fold, Y_val_fold = (
                    self.Y_train[train_idx],
                    self.Y_train[val_idx],
                )
                E_train_fold, E_val_fold = (
                    self.E_train[train_idx],
                    self.E_train[val_idx],
                )

                if const_idxs:
                    X_inv = X_train_fold @ (
                        U[:, const_idxs] @ U[:, const_idxs].T
                    )
                    rf = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_features=self.max_features,
                    )
                    rf.fit(X_inv, Y_train_fold)
                    # TODO: check if you have to project also to make predictions.
                    preds = rf.predict(
                        X_val_fold @ (U[:, const_idxs] @ U[:, const_idxs].T)
                    )
                else:
                    preds = np.zeros((len(Y_val_fold)))

                residuals = Y_val_fold - preds
                rss_per_env = [
                    np.mean((residuals[E_val_fold == e]) ** 2)
                    for e in np.unique(E_val_fold)
                ]
                var_th[th_idx, fd_idx] = np.var(rss_per_env)

            # Average over folds
            var[th_idx] = np.mean(var_th[th_idx, :])

        th_opt = th_const[np.argmin(var)]

        self.var_th = var_th
        self.var = var
        self.blocks_shape = blocks_shape
        self.th_const = th_const
        self.th_opt = th_opt

    def predict_isd(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict a response given feature values.

        Args:
            X: Feature values.

        Returns:
            preds: predictions made using the invariant subspace.
        """
        const_blocks, const_idxs, v_idxs = self._check_const(
            self.blocks_shape, self.th_const, self.th_opt
        )
        if const_idxs:
            X_inv = self.X_train @ (
                self.U[:, const_idxs] @ self.U[:, const_idxs].T
            )
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(X_inv, self.Y_train)
            preds = rf.predict(
                X @ (self.U[:, const_idxs] @ self.U[:, const_idxs].T)
            )
        else:
            preds = np.zeros((X.shape[0]))

        return preds
