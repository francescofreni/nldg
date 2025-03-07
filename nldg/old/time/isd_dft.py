from __future__ import division
import numpy as np
import statsmodels.api as sm
from nldg.utils.jbd import jbd, ajbd
from cvxopt.solvers import qp
from cvxopt import matrix


class ISD:

    def __init__(self, X_hist, Y_hist, w_size=None):
        self.X = X_hist
        self.Y = Y_hist
        self.n = X_hist.shape[0]
        self.p = X_hist.shape[1]
        if not w_size:
            w_size = [2 * self.p] * np.ceil(self.n / (2 * self.p))
        self.w = w_size
        self.m = len(w_size)
        self.w_st = np.linspace(
            0, self.n - self.w[-1], num=self.m, endpoint=True, dtype=int
        )
        self.w_end = self.w_st + self.w

        # estimates initialization
        self.beta_inv = np.zeros((self.p, 1))
        self.beta_pooled = None
        self.beta_maximin = None
        self.beta_ols = None

        self.blocks = list([self.p])
        self.c_blocks = None
        self.c_idxs = list([self.p])
        self.v_idxs = list([self.p])
        self.U = np.zeros((self.p, self.p))

    def invariant_estimator(self, k_fold=None, diag=False, std=True):
        # estimator for the invariant component beta_inv
        # k_fold: number of folds for CV. If None, LOO CV is run
        # diag: True if Sigmas are assumed to be jointly diagonalizable
        # std: if True, invariance threshold takes into account standard error

        # auxiliary functions
        def comp_thresholds(X, Y, w, mu):
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
                Y_w = Y[w_start[k] : w_start[k] + w[k], :]
                cov_w = np.cov(Y_w - X_w @ mu, X_w @ mu, rowvar=False)
                c[k] = cov_w[0, 1]
                v[k] = np.sqrt(cov_w[0, 0] * cov_w[1, 1])

            T = np.mean([np.abs((c[k]) / v[k]) for k in range(m)])
            return T

        def check_const(blocks_shape, th_const, th):
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

        def xpl_var(X, Y, beta):
            # compute explained varinace by beta (under zero mean ass.)
            # X, Y: observed covariates and response
            # beta: linear parameter
            if len(X.shape) == 1:
                n = 1
                X = X.reshape(1, -1)
                Y = Y.reshape(1, -1)
            else:
                n = X.shape[0]
            return (1 / n) * (2 * Y.T @ X @ beta - beta.T @ X.T @ X @ beta)

        # 1) Compute covariance and coefficients estimates
        Sigma = np.zeros((self.m, self.p, self.p))
        gamma = np.zeros((self.m, self.p + 1, 1))

        for idx, win in enumerate(self.w):
            X_w = self.X[self.w_st[idx] : self.w_st[idx] + win, :]
            Y_w = self.Y[self.w_st[idx] : self.w_st[idx] + win, :]
            gamma[idx, :, :] = self.ols(Y_w, X_w)
            Sigma[idx, :, :] = np.cov(X_w, rowvar=False)

        # Save pooled regression coefficient
        beta_pooled = np.mean(gamma, axis=0)
        self.beta_pooled = beta_pooled[1:]
        beta_icpt = beta_pooled[0]

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

        # Transform estimated coefficients using JBD U
        # and check if they are constant
        gamma_til = np.zeros((self.m, self.p, 1))
        for k in range(self.m):
            gamma_til[k, :, :] = U @ gamma[k, 1:, :]

        # Compute constant thresholds for every estimated block
        th_const = []
        for b, bs in enumerate(blocks_shape):
            if b == 0:
                block_idxs = list(range(bs))
            else:
                block_idxs = [j + sum(blocks_shape[:b]) for j in range(bs)]
            gamma_block = gamma_til[:, block_idxs, :]
            mu_block = np.mean(gamma_block, axis=0)
            th_const.append(
                comp_thresholds(
                    self.X @ (U.T[:, block_idxs]), self.Y, self.w, mu_block
                )
            )
        th_const.append(1)

        # 3) Ivariant parameter estimation
        #  Cross-validation for threshold selection
        xi_inv = np.zeros((self.p, 1))
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
            const_blocks, const_idxs, v_idxs = check_const(
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
                Y_ts = self.Y[wst_fold[fd_idx] : wst_fold[fd_idx] + w_fold, :]

                # Compute candidate invariant parameter
                if const_idxs:
                    X_inv = X_tr @ (U.T[:, const_idxs])
                    xi_inv = self.ols(Y_tr, X_inv)[1:]
                    beta_inv = U.T[:, const_idxs] @ xi_inv
                else:
                    beta_inv = np.zeros((self.p, 1))

                # Compute empirical explained variance by adapted
                # invariant parameter on test fold
                for t in range(T_test):
                    X_test = X_ts[t : t + ws_test, :]
                    Y_test = Y_ts[t : t + ws_test, :]
                    X_val = X_ts[t + ws_test : t + ws_test + 1, :]
                    Y_val = Y_ts[t + ws_test : t + ws_test + 1, :]
                    delta_res = self.adapt(X_test, Y_test, beta_inv, v_idxs)
                    xpl_v_fd[t] = xpl_var(X_val, Y_val, delta_res + beta_inv)

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

        # Compute invariant parameter for optimal threshold
        xi_inv = np.zeros((self.p, 1))
        const_blocks, const_idxs, v_idxs = check_const(
            blocks_shape, th_const, th_opt
        )
        if const_idxs:
            X_inv = self.X @ (U.T[:, const_idxs])
            xi_inv = self.ols(self.Y, X_inv)[1:]
            beta_inv = U.T[:, const_idxs] @ xi_inv
        else:
            beta_inv = np.zeros((self.p, 1))

        return beta_inv, beta_icpt, U, blocks_shape, const_blocks

    def ols(self, Y, X, icpt=True):
        # Computes OLS solution regressing Y onto X
        if icpt:
            X = sm.add_constant(X)
        return sm.OLS(Y, X).fit().params.reshape(-1, 1)

    def adapt(self, X_ad, Y_ad, beta_inv=None, v_idxs=None):
        # Adaptation step
        # X_ad: observed covariates in adaptation window
        # Y_ad: observed response in adaptation window
        # beta_inv: invariant component. If None, self.beta_inv is used
        # v_idxs: list of indexes corresponding to residual subspace (wrt U)
        # returns: estimated residual component delta_res

        if beta_inv is None:
            beta_inv = self.beta_inv
        if v_idxs is None:
            v_idxs = self.v_idxs

        if v_idxs:
            if len(v_idxs) < self.p:
                Y_ad_res = Y_ad - X_ad @ beta_inv
                X_ad_res = X_ad @ (self.U.T)
                X_ad_res = X_ad_res[:, v_idxs]
                delta_res = (
                    self.U.T[:, v_idxs] @ self.ols(Y_ad_res, X_ad_res)[1:]
                )
            else:
                delta_res = self.ols(Y_ad, X_ad)[1:]
        else:
            delta_res = np.zeros((self.p, 1))

        return delta_res

    def magging_estimator(self):
        # Implementation of the maximin effect following the pseudocode
        # provided in the reference paper
        # Reference: Bühlmann, P., & Meinshausen, N. (2015).
        #            Magging: maximin aggregation for inhomogeneous
        #            large-scale data.
        #            Proceedings of the IEEE, 104(1), 126-135.

        beta_hat_g_list = []
        w_start = self.w_st
        for g, gs in enumerate(self.w):
            X_g = self.X[w_start[g] : w_start[g] + gs, :]
            X_1_g = np.concatenate((X_g, np.ones((gs, 1))), axis=1)
            Y_g = self.Y[w_start[g] : w_start[g] + gs, :]
            beta_hat_g_list.append(sm.OLS(Y_g, X_1_g).fit().params[:-1])

        beta_hat = np.array(beta_hat_g_list).T
        g_num = beta_hat.shape[1]

        Sigma_hat = (1 / self.n) * self.X.T @ self.X

        H = beta_hat.T @ Sigma_hat @ beta_hat
        H = matrix(H, (g_num, g_num))
        A = matrix(
            np.concatenate((np.ones((1, g_num)), np.eye(g_num)), axis=0),
            (g_num + 1, g_num),
        )

        B = np.zeros((g_num + 1, 1))
        B[0, 0] = 1
        B = matrix(B)
        Q = matrix(np.zeros((g_num, 1)))

        weights = qp(H, Q, -A, -B, options={"show_progress": False})
        w = np.array(weights["x"])

        self.beta_maximin = beta_hat @ w
        return self.beta_maximin

    def get_pooled_est(self):
        # Returns pooled OLS solution computed by invariant_est()
        return self.beta_pooled
