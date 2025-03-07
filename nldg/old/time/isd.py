"""
Code inspired from https://github.com/mlazzaretto/Invariant-Subspace-Decomposition.git
"""

from __future__ import division
import numpy as np
from nldg.utils.jbd import jbd, ajbd
from sklearn.ensemble import RandomForestRegressor


class IsdRF:

    def __init__(
        self,
        X_hist,
        Y_hist,
        w_size=None,
        n_estimators=50,
        random_state=42,
        max_features=1.0,
    ):
        self.X = X_hist
        self.Y = Y_hist
        self.n = X_hist.shape[0]
        self.p = X_hist.shape[1]
        if not w_size:
            w_size = [2 * self.p] * int(self.n / (2 * self.p))
        self.w = w_size
        self.m = len(w_size)
        self.w_st = np.linspace(
            0, self.n - self.w[-1], num=self.m, endpoint=True, dtype=int
        )
        self.w_end = self.w_st + self.w

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
        self.xpl_v_th = None
        self.xpl_v = None
        self.xpl_v_min = None
        self.rf_inv = None

    def check_const(self, blocks_shape, th_const, th):
        # check which blocks are invariant for a given threshold
        # blocks shape: list of blocks dimensions
        # th_const: list of thresholds corresponding to each block
        # th: reference threshold to check
        # returns:
        #   const_blocks: boolean vector for constant blocks
        #   const_idxs: list of covariates indices corresponding
        #               to constant blocks
        #   v_idxs: list of covariates indices corresponding
        #           to time-varying blocks
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

    def find_invariant(self, k_fold=None, diag=False, std=True):
        # estimator for the invariant component beta_inv
        # k_fold: number of folds for CV. If None, LOO CV is run
        # diag: True if Sigmas are assumed to be jointly diagonalizable
        # std: if True, invariance threshold takes into account standard error

        # auxiliary functions
        def comp_thresholds(X, Y, w, rf):
            # compute invariance threshold T for constant coefficient selection
            # for a given block
            # X, Y: observed predictors and responses
            # w: list of window dimensions over which mu is computed
            # mu: mean coefficient value

            m = len(w)
            n = X.shape[0]
            w_start = np.linspace(
                0, n - w[-1], num=m, endpoint=True, dtype=int
            )
            c = np.zeros((m,))
            v = np.zeros((m,))
            for k in range(m):
                X_w = X[w_start[k] : w_start[k] + w[k], :]
                Y_w = Y[w_start[k] : w_start[k] + w[k]]
                fitted_values = rf.predict(X_w)
                cov_w = np.cov(
                    Y_w - fitted_values, fitted_values, rowvar=False
                )
                c[k] = cov_w[0, 1]
                v[k] = np.sqrt(cov_w[0, 0] * cov_w[1, 1])

            T = np.mean([np.abs((c[k]) / v[k]) for k in range(m)])
            return T

        def xpl_var(Y, Yhat):
            # compute explained variance by beta (under zero mean ass.)
            # X, Y: observed covariates and response
            # beta: linear parameter
            n = len(Y)
            return (1 / n) * (2 * Y.T @ Yhat - Yhat.T @ Yhat)

        # 1) Compute covariance and coefficients estimates
        Sigma = np.zeros((self.m, self.p, self.p))

        for idx, win in enumerate(self.w):
            X_w = self.X[self.w_st[idx] : self.w_st[idx] + win, :]
            Sigma[idx, :, :] = np.cov(X_w, rowvar=False)

        # 2) Joint block diagonalization
        if diag:
            U, _, _, _ = jbd(Sigma, threshold=0, diag=True)
            blocks_shape = list[np.ones(self.p)]
            Sigma_diag = np.zeros_like(Sigma)
            for k in range(self.m):
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

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(self.X @ (U.T[:, block_idxs]), self.Y)
            th_const.append(
                comp_thresholds(
                    self.X @ (U.T[:, block_idxs]),
                    self.Y,
                    self.w,
                    rf,
                )
            )
        th_const.append(1)

        # 3) Ivariant parameter estimation
        #  Cross-validation for threshold selection
        if not k_fold:
            k_fold = self.m
        w_fold = int(self.n / k_fold)
        wst_fold = np.linspace(
            0, self.n - w_fold, num=k_fold, endpoint=True, dtype=int
        )
        ws_test = 2 * self.p
        xpl_v = np.zeros(len(th_const))
        xpl_v_th = np.zeros((len(th_const), k_fold))
        for th_idx, th in enumerate(th_const):
            # Detect constant blocks for threshold th
            const_blocks, const_idxs, v_idxs = self.check_const(
                blocks_shape, th_const, th
            )
            for fd_idx in range(k_fold):
                # Leave out one fold for testing
                T_test = w_fold - ws_test - 1
                xpl_v_fd = np.zeros(T_test)
                X_tr = np.delete(
                    self.X,
                    np.s_[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold],
                    axis=0,
                )
                Y_tr = np.delete(
                    self.Y,
                    np.s_[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold],
                    axis=0,
                )
                X_ts = self.X[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold, :]
                Y_ts = self.Y[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold]

                if const_idxs:
                    X_inv = X_tr @ (U.T[:, const_idxs])
                    rf_inv = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_features=self.max_features,
                    )
                    rf_inv.fit(X_inv, Y_tr)

                # Compute empirical explained variance by adapted
                # invariant parameter on test fold
                for t in range(T_test):
                    X_test = X_ts[t : t + ws_test, :]
                    Y_test = Y_ts[t : t + ws_test]
                    X_val = X_ts[t + ws_test : t + ws_test + 1, :]
                    Y_val = Y_ts[t + ws_test : t + ws_test + 1]

                    if const_idxs:
                        fitted_inv = rf_inv.predict(
                            X_test @ (U.T[:, const_idxs])
                        )
                        fitted_res = self.adapt(
                            X_test, Y_test, fitted_inv, X_val, v_idxs
                        )
                        fitted_inv = rf_inv.predict(
                            X_val @ (U.T[:, const_idxs])
                        )
                        Yhat = fitted_inv + fitted_res
                    else:
                        fitted_inv = np.zeros((X_test.shape[0],))
                        fitted_res = self.adapt(
                            X_test, Y_test, fitted_inv, X_val, v_idxs
                        )
                        Yhat = fitted_res
                    xpl_v_fd[t] = xpl_var(Y_val, Yhat)

                xpl_v_th[th_idx, fd_idx] = np.mean(xpl_v_fd)
            # Average over folds
            xpl_v[th_idx] = np.mean(xpl_v_th[th_idx, :])
        # Standard error across folds for all thresholds
        std_th = (1 / np.sqrt(k_fold)) * np.std(xpl_v_th, axis=1)
        # Minimum explained variance across folds for all thresholds
        xpl_v_min = np.min(xpl_v_th, axis=1)

        # print("expl. var", xpl_v, " std th", std_th)

        # Optimal threshold selection
        if np.all(xpl_v_min[:-1] <= 0):
            th_opt = 0
        else:
            sort_th = np.argsort(th_const)
            xpl_v_s = xpl_v[sort_th]
            xv_max_idx = np.argmax(xpl_v_s)
            xv_max = xpl_v_s[xv_max_idx]
            if std:
                xv_min = xv_max - std_th[sort_th][xv_max_idx]
            else:
                xv_min = xv_max
            th_cand = np.where(xpl_v_s[0 : xv_max_idx + 1] >= xv_min)[0][0]
            th_opt = np.array(th_const)[sort_th][th_cand]

        self.xpl_v_th = xpl_v_th
        self.xpl_v = xpl_v
        self.xpl_v_min = xpl_v_min
        self.blocks_shape = blocks_shape
        self.th_const = th_const
        self.th_opt = th_opt

        const_blocks, const_idxs, v_idxs = self.check_const(
            self.blocks_shape, self.th_const, self.th_opt
        )
        if const_idxs:
            X_inv = self.X @ (self.U.T[:, const_idxs])
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(X_inv, self.Y)
            self.rf_inv = rf

    def adapt(self, X_ad, Y_ad, Y_ad_hat, X_val, v_idxs=None):
        # Adaptation step
        # X_ad: observed covariates in adaptation window
        # Y_ad: observed response in adaptation window
        # beta_inv: invariant component. If None, self.beta_inv is used
        # v_idxs: list of indexes corresponding to residual subspace (wrt U)
        # returns: estimated residual component delta_res

        if v_idxs is None:
            v_idxs = self.v_idxs

        if v_idxs:
            if len(v_idxs) < self.p:
                Y_ad_res = Y_ad - Y_ad_hat
                X_ad_res = X_ad @ (self.U.T)
                X_ad_res = X_ad_res[:, v_idxs]
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_features=self.max_features,
                )
                rf.fit(X_ad_res, Y_ad_res)
                fitted_res = rf.predict(X_val @ (self.U.T[:, v_idxs]))
            else:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_features=self.max_features,
                )
                rf.fit(X_ad, Y_ad)
                fitted_res = rf.predict(X_val @ (self.U.T))
        else:
            fitted_res = np.zeros((X_val.shape[0],))

        return fitted_res

    def predict_zeroshot(
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
        const_blocks, const_idxs, v_idxs = self.check_const(
            self.blocks_shape, self.th_const, self.th_opt
        )
        if const_idxs:
            preds = self.rf_inv.predict(X @ (self.U.T[:, const_idxs]))
        else:
            preds = np.zeros((X.shape[0]))

        return preds


class IsdRF2:

    def __init__(
        self,
        X_hist,
        Y_hist,
        w_size=None,
        n_estimators=50,
        random_state=42,
        max_features=1.0,
    ):
        self.X = X_hist
        self.Y = Y_hist
        self.n = X_hist.shape[0]
        self.p = X_hist.shape[1]
        if not w_size:
            w_size = [2 * self.p] * int(self.n / (2 * self.p))
        self.w = w_size
        self.m = len(w_size)
        self.w_st = np.linspace(
            0, self.n - self.w[-1], num=self.m, endpoint=True, dtype=int
        )
        self.w_end = self.w_st + self.w

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
        self.xpl_v_th = None
        self.xpl_v = None
        self.xpl_v_min = None
        self.rf_inv = None

    def check_const(self, blocks_shape, th_const, th):
        # check which blocks are invariant for a given threshold
        # blocks shape: list of blocks dimensions
        # th_const: list of thresholds corresponding to each block
        # th: reference threshold to check
        # returns:
        #   const_blocks: boolean vector for constant blocks
        #   const_idxs: list of covariates indices corresponding
        #               to constant blocks
        #   v_idxs: list of covariates indices corresponding
        #           to time-varying blocks
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

    def find_invariant(self, k_fold=None, diag=False, std=True):
        # estimator for the invariant component beta_inv
        # k_fold: number of folds for CV. If None, LOO CV is run
        # diag: True if Sigmas are assumed to be jointly diagonalizable
        # std: if True, invariance threshold takes into account standard error

        # auxiliary functions
        def comp_thresholds(X, Y, w, rf):
            # compute invariance threshold T for constant coefficient selection
            # for a given block
            # X, Y: observed predictors and responses
            # w: list of window dimensions over which mu is computed
            # mu: mean coefficient value

            m = len(w)
            n = X.shape[0]
            w_start = np.linspace(
                0, n - w[-1], num=m, endpoint=True, dtype=int
            )
            c = np.zeros((m,))
            v = np.zeros((m,))
            for k in range(m):
                X_w = X[w_start[k] : w_start[k] + w[k], :]
                Y_w = Y[w_start[k] : w_start[k] + w[k]]
                fitted_values = rf.predict(X_w)
                cov_w = np.cov(
                    Y_w - fitted_values, fitted_values, rowvar=False
                )
                c[k] = cov_w[0, 1]
                v[k] = np.sqrt(cov_w[0, 0] * cov_w[1, 1])

            T = np.mean([np.abs((c[k]) / v[k]) for k in range(m)])
            return T

        def xpl_var(Y, Yhat):
            # compute explained variance by beta (under zero mean ass.)
            # X, Y: observed covariates and response
            # beta: linear parameter
            n = len(Y)
            return (1 / n) * (2 * Y.T @ Yhat - Yhat.T @ Yhat)

        # 1) Compute covariance and coefficients estimates
        Sigma = np.zeros((self.m, self.p, self.p))

        for idx, win in enumerate(self.w):
            X_w = self.X[self.w_st[idx] : self.w_st[idx] + win, :]
            Sigma[idx, :, :] = np.cov(X_w, rowvar=False)

        # 2) Joint block diagonalization
        if diag:
            U, _, _, _ = jbd(Sigma, threshold=0, diag=True)
            blocks_shape = list[np.ones(self.p)]
            Sigma_diag = np.zeros_like(Sigma)
            for k in range(self.m):
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

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(self.X[:, block_idxs], self.Y)
            th_const.append(
                comp_thresholds(
                    self.X[:, block_idxs],
                    self.Y,
                    self.w,
                    rf,
                )
            )
        th_const.append(1)

        # 3) Ivariant parameter estimation
        #  Cross-validation for threshold selection
        if not k_fold:
            k_fold = self.m
        w_fold = int(self.n / k_fold)
        wst_fold = np.linspace(
            0, self.n - w_fold, num=k_fold, endpoint=True, dtype=int
        )
        ws_test = 2 * self.p
        xpl_v = np.zeros(len(th_const))
        xpl_v_th = np.zeros((len(th_const), k_fold))
        for th_idx, th in enumerate(th_const):
            # Detect constant blocks for threshold th
            const_blocks, const_idxs, v_idxs = self.check_const(
                blocks_shape, th_const, th
            )
            for fd_idx in range(k_fold):
                # Leave out one fold for testing
                T_test = w_fold - ws_test - 1
                xpl_v_fd = np.zeros(T_test)
                X_tr = np.delete(
                    self.X,
                    np.s_[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold],
                    axis=0,
                )
                Y_tr = np.delete(
                    self.Y,
                    np.s_[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold],
                    axis=0,
                )
                X_ts = self.X[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold, :]
                Y_ts = self.Y[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold]

                if const_idxs:
                    X_inv = X_tr[:, const_idxs]
                    rf_inv = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_features=self.max_features,
                    )
                    rf_inv.fit(X_inv, Y_tr)

                # Compute empirical explained variance by adapted
                # invariant parameter on test fold
                for t in range(T_test):
                    X_test = X_ts[t : t + ws_test, :]
                    Y_test = Y_ts[t : t + ws_test]
                    X_val = X_ts[t + ws_test : t + ws_test + 1, :]
                    Y_val = Y_ts[t + ws_test : t + ws_test + 1]

                    if const_idxs:
                        fitted_inv = rf_inv.predict(X_test[:, const_idxs])
                        fitted_res = self.adapt(
                            X_test, Y_test, fitted_inv, X_val, v_idxs
                        )
                        fitted_inv = rf_inv.predict(X_val[:, const_idxs])
                        Yhat = fitted_inv + fitted_res
                    else:
                        fitted_inv = np.zeros((X_test.shape[0],))
                        fitted_res = self.adapt(
                            X_test, Y_test, fitted_inv, X_val, v_idxs
                        )
                        Yhat = fitted_res
                    xpl_v_fd[t] = xpl_var(Y_val, Yhat)

                xpl_v_th[th_idx, fd_idx] = np.mean(xpl_v_fd)
            # Average over folds
            xpl_v[th_idx] = np.mean(xpl_v_th[th_idx, :])
        # Standard error across folds for all thresholds
        std_th = (1 / np.sqrt(k_fold)) * np.std(xpl_v_th, axis=1)
        # Minimum explained variance across folds for all thresholds
        xpl_v_min = np.min(xpl_v_th, axis=1)

        # print("expl. var", xpl_v, " std th", std_th)

        # Optimal threshold selection
        if np.all(xpl_v_min[:-1] <= 0):
            th_opt = 0
        else:
            sort_th = np.argsort(th_const)
            xpl_v_s = xpl_v[sort_th]
            xv_max_idx = np.argmax(xpl_v_s)
            xv_max = xpl_v_s[xv_max_idx]
            if std:
                xv_min = xv_max - std_th[sort_th][xv_max_idx]
            else:
                xv_min = xv_max
            th_cand = np.where(xpl_v_s[0 : xv_max_idx + 1] >= xv_min)[0][0]
            th_opt = np.array(th_const)[sort_th][th_cand]

        self.xpl_v_th = xpl_v_th
        self.xpl_v = xpl_v
        self.xpl_v_min = xpl_v_min
        self.blocks_shape = blocks_shape
        self.th_const = th_const
        self.th_opt = th_opt

        const_blocks, const_idxs, v_idxs = self.check_const(
            self.blocks_shape, self.th_const, self.th_opt
        )
        if const_idxs:
            X_inv = self.X[:, const_idxs]
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            rf.fit(X_inv, self.Y)
            self.rf_inv = rf

    def adapt(self, X_ad, Y_ad, Y_ad_hat, X_val, v_idxs=None):
        # Adaptation step
        # X_ad: observed covariates in adaptation window
        # Y_ad: observed response in adaptation window
        # beta_inv: invariant component. If None, self.beta_inv is used
        # v_idxs: list of indexes corresponding to residual subspace (wrt U)
        # returns: estimated residual component delta_res

        if v_idxs is None:
            v_idxs = self.v_idxs

        if v_idxs:
            if len(v_idxs) < self.p:
                Y_ad_res = Y_ad - Y_ad_hat
                X_ad_res = X_ad[:, v_idxs]
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_features=self.max_features,
                )
                rf.fit(X_ad_res, Y_ad_res)
                fitted_res = rf.predict(X_val[:, v_idxs])
            else:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_features=self.max_features,
                )
                rf.fit(X_ad, Y_ad)
                fitted_res = rf.predict(X_val)
        else:
            fitted_res = np.zeros((X_val.shape[0],))

        return fitted_res

    def predict_zeroshot(
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
        const_blocks, const_idxs, v_idxs = self.check_const(
            self.blocks_shape, self.th_const, self.th_opt
        )
        if const_idxs:
            preds = self.rf_inv.predict(X[:, const_idxs])
        else:
            preds = np.zeros((X.shape[0]))

        return preds
