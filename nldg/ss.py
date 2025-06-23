import numpy as np
import cvxpy as cp
from scipy.interpolate import BSpline
from scipy.stats import iqr
import torch
from typing import Callable, Optional
import warnings


class MinimaxSmoothSpline:
    """
    Fit a smoothing spline to (x, y) data using either empirical risk minimization (ERM)
    or a minimax strategy over environments.

    Parameters
    ----------
    x : array-like
        Predictor variable values. Must be finite and 1-dimensional.
    y : array-like
        Response variable values. Must be finite and of the same length as `x`.
    env : array-like, optional (required if method="minimax")
        Environment labels corresponding to each (x, y) point. Used only for
        the "minimax" method to define groups over which to minimize the
        maximum mean squared error.
    w : array-like, optional
        Observation weights. Must be non-negative. If None, uniform weights are used.
    degree : int, optional (default=3)
        Degree of the B-spline basis. Common choices are 1 (linear), 2 (quadratic), 3 (cubic).
    lam : float, optional (default=None)
        Smoothing parameter controlling the trade-off between fidelity to data
        and smoothness (penalized second derivative).
    cv: bool (default=False)
        If True, performs ordinary leave-one-out cross-validation when method is `erm`,
        and 5-fold cross-validation if method is `minimax`.
        If cv is True and lambda is not None, lambda is overwritten after performing LOOCV.
    all_knots : bool, optional (default=False)
        If True, uses all unique x values as internal knots. If False, selects
        a reduced number of knots using `nknots_func`.
    nknots_func : callable or int, optional
        Function or integer to determine the number of internal knots if
        `all_knots` is False. Defaults to a heuristic based on data size.
    tol : float, optional
        Tolerance for merging similar x values. Defaults to `1e-6 * IQR(x)`.
    method : {"erm", "minimax"}, optional (default="minimax")
        Estimation method:
            - "erm": Empirical Risk Minimization (standard smoothing spline)
            - "minimax": Minimizes the worst-case environment-specific MSE
    opt_method : {"cp", "extragradient"}, optional (default="cp")
        Optimization method for minimax objective:
            - "cp": Convex program using CVXPY
            - "extragradient": Gradient-based saddle point optimization

    **kwargs :
        Additional arguments passed to extragradient optimizer if `opt_method="extragradient"`,
        e.g., `alpha`, `epochs`, `verbose`, etc.

    Attributes
    ----------
    coef : ndarray
        Coefficients of the fitted B-spline basis.
    spline : BSpline object
        Scipy BSpline object for evaluating the spline.
    ux : ndarray
        Unique x values after preprocessing.
    knots : ndarray
        Knot sequence for the B-spline basis.
    Omega : ndarray
        Penalty matrix used in smoothness regularization.
    x_min, x_max : float
        Domain range for the fitted spline.

    Methods
    -------
    predict(x_new)
        Evaluate the fitted spline at new data points `x_new`.

    Notes
    -----
    - This implementation generalizes `smooth.spline` to a robust minimax setting.
    - `cp` may be often preferred over `extragradient`.
    - Spline fitting uses second-derivative penalization similar to classical smoothing splines.
    - Cross-validation with the minimax approach may be computational expensive and not all values of
      lambda can be used: the second-derivative matrix is ill-conditioned if the number of basis functions
      is large.

    Examples
    --------
    >>> spline = MinimaxSmoothSpline(x, y, env=env_labels, method="minimax")
    >>> y_pred = spline.predict(x_new)

    >>> spline_erm = MinimaxSmoothSpline(x, y, method="erm")
    >>> y_smooth = spline_erm.predict(x_new)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        env: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        degree: int = 3,
        lam: Optional[float] = None,
        cv: bool = False,
        all_knots: bool = False,
        nknots_func: Optional[Callable[[int], int]] = None,
        tol: Optional[float] = None,
        method: str = "minimax",
        opt_method: str = "cp",
        **kwargs,
    ):
        if method not in ["erm", "minimax"]:
            raise ValueError("method must be 'erm' or 'minimax'")
        self.method = method

        if self.method == "minimax" and env is None:
            raise ValueError("env must not be None if method is minimax")

        if opt_method not in ["cp", "extragradient"]:
            raise ValueError("opt_method must be 'cp' or 'extragradient'")
        self.opt_method = opt_method

        if self.method == "minimax":
            env = np.asarray(env).flatten()

        self.degree = degree
        self.lam = lam
        self.cv = cv

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(x)
        if len(y) != n:
            raise ValueError("x and y must have same length")

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("x and y must be finite")

        # Weights
        if w is None:
            w = np.ones(n)
        else:
            w = np.asarray(w)
            if len(w) != n:
                raise ValueError("x and w must have same length")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            if np.all(w == 0):
                raise ValueError("at least one weight must be positive")
            w = w * np.sum(w > 0) / np.sum(w)

        # Tolerance
        if tol is None:
            tol = 1e-6 * iqr(x)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("'tol' must be strictly positive and finite")

        # Group by unique x (rounded)
        xx = np.round((x - np.mean(x)) / tol)
        _, idx_first = np.unique(xx, return_index=True)
        ux = np.sort(x[idx_first])
        uxx = np.sort(xx[idx_first])
        nx = len(ux)

        if nx <= 3:
            raise ValueError("need at least four unique x values")

        if nx == n:
            ox = np.argsort(x)
            tmp = np.column_stack([w, w * y, w * y**2])[ox]
            if self.method == "minimax":
                self.envbar = env[ox]
        else:
            ox = np.searchsorted(uxx, xx)
            tmp = np.zeros((nx, 3))
            if self.method == "minimax":
                envbar = np.empty(nx, dtype=env.dtype)
            for i in range(nx):
                mask = ox == i
                wi = w[mask]
                yi = y[mask]
                tmp[i, 0] = np.sum(wi)
                tmp[i, 1] = np.sum(wi * yi)
                tmp[i, 2] = np.sum(wi * yi**2)
                if self.method == "minimax":
                    vals, counts = np.unique(env[mask], return_counts=True)
                    envbar[i] = vals[np.argmax(counts)]
            if self.method == "minimax":
                self.envbar = envbar

        self.wbar = tmp[:, 0]
        self.ybar = tmp[:, 1] / np.where(self.wbar > 0, self.wbar, 1)
        self.yssw = np.sum(tmp[:, 2] - self.wbar * self.ybar**2)

        # Store original domain info
        self.ux = ux
        self.x_min = ux.min()
        self.x_max = ux.max()
        # self.r_ux = ux[-1] - ux[0]
        # self.xbar = (ux - ux[0]) / self.r_ux

        # Knots - uniform spacing
        if all_knots:
            nknots = nx
        else:
            if nknots_func is None:
                nknots = self._nknots_smspl(nx)
            elif callable(nknots_func):
                nknots = nknots_func(nx)
            elif isinstance(nknots_func, int):
                nknots = nknots_func
            else:
                raise ValueError("'nknots' must be numeric or a callable")

            if nknots < 1:
                raise ValueError("'nknots' must be at least 1")
            elif nknots > nx:
                raise ValueError(
                    "Cannot use more inner knots than unique x values"
                )

        # Create uniform internal knots
        if all_knots:
            inner_knots = ux
        else:
            inner_knots = np.linspace(self.x_min, self.x_max, nknots)

        self.knots = np.concatenate(
            [
                np.repeat(self.x_min, self.degree),
                inner_knots,
                np.repeat(self.x_max, self.degree),
            ]
        )
        self.M = len(self.knots) - self.degree - 1

        self._fit(**kwargs)

    @staticmethod
    def _nknots_smspl(n: int):
        if n < 50:
            return n
        else:
            a1 = np.log2(50)
            a2 = np.log2(100)
            a3 = np.log2(140)
            a4 = np.log2(200)
            if n < 200:
                return int(2 ** (a1 + (a2 - a1) * (n - 50) / 150))
            elif n < 800:
                return int(2 ** (a2 + (a3 - a2) * (n - 200) / 600))
            elif n < 3200:
                return int(2 ** (a3 + (a4 - a3) * (n - 800) / 2400))
            else:
                return int(200 + (n - 3200) ** 0.2)

    def _bspline_dm(self, x_vals: np.ndarray):
        X = np.zeros((len(x_vals), self.M))
        coeffs = np.eye(self.M)
        for i in range(self.M):
            b = BSpline(self.knots, coeffs[i], self.degree, extrapolate=False)
            X[:, i] = b(x_vals)
        return X

    def _omega(self, grid: np.ndarray):
        M = self.M

        # Precompute all second derivatives at once
        second_derivs = np.empty((M, len(grid)))
        for i in range(M):
            coeff = np.zeros(M)
            coeff[i] = 1
            b = BSpline(
                self.knots, coeff, self.degree, extrapolate=False
            ).derivative(2)
            second_derivs[i] = b(grid)

        # Vectorized trapezoidal integration
        dx = grid[1] - grid[0]
        trap_weights = np.ones(len(grid))
        trap_weights[0] = trap_weights[-1] = 0.5
        weighted_derivs = second_derivs * trap_weights[np.newaxis, :]
        Omega = dx * (weighted_derivs @ second_derivs.T)

        return Omega

    @staticmethod
    def _project_onto_simplex(v: np.ndarray) -> np.ndarray:
        """
        Projection onto the probability simplex.
        Reference: Wang et al. (2013).
            "Projection onto the probability simplex:
            An efficient algorithm with a simple proof, and an application"
            https://arxiv.org/pdf/1309.1541
        """
        original_shape = v.shape
        v_flat = v.flatten()
        D = v_flat.size

        # Step 1: Sort in descending order
        u = np.sort(v_flat)[::-1]

        # Step 2: Find rho
        cssv = np.cumsum(u)
        j = np.arange(1, D + 1)
        condition = u + (1.0 / j) * (1 - cssv)
        rho = np.where(condition > 0)[0].max() + 1

        # Step 3: Compute lambda
        lambda_val = (1 - np.sum(u[:rho])) / rho

        # Step 4: Compute projection
        x = np.maximum(v_flat + lambda_val, 0)

        return x.reshape(original_shape)

    def _train_extragradient(
        self,
        alpha: float = 0.5,
        epochs: int = 1000,
        seed: int = 42,
        verbose: bool = False,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        x = np.asarray(self.ux).flatten()
        y = np.asarray(self.ybar).flatten()
        env = np.asarray(self.envbar).flatten()
        w = np.asarray(self.wbar).flatten()

        E_unique = np.unique(env)
        E = len(E_unique)

        # Prepare design matrix function
        N_full = self._bspline_dm(x)
        N_full_t = torch.tensor(N_full, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        w_t = torch.tensor(w, dtype=torch.float64)
        env_t = torch.tensor(env, dtype=torch.long)
        Omega = self.Omega
        Omega = Omega / np.linalg.norm(Omega, ord="fro")
        Omega_t = torch.tensor(Omega, dtype=torch.float64)

        beta = torch.zeros(self.M, dtype=torch.float64)
        p = torch.ones(E, dtype=torch.float64) / E

        best_max_loss = np.inf
        epochs_no_improvement = 0
        for t in range(epochs):
            # Compute losses and gradients at current point
            losses = []
            grad = torch.zeros_like(beta)
            for i, e in enumerate(E_unique):
                idx = env_t == e
                N_e = N_full_t[idx]
                y_e = y_t[idx]
                w_e = w_t[idx]
                pred = N_e @ beta
                residual = pred - y_e
                loss = torch.mean(w_e * residual**2)
                losses.append(loss)
                grad_e = (2.0 / idx.sum()) * (N_e.T @ (w_e * residual))
                grad += p[i] * grad_e
            losses = torch.stack(losses)
            grad += 2 * self.lam * (Omega_t @ beta)

            # Extragradient step 1: half-step
            beta_half = beta - alpha * grad
            p_half = torch.tensor(
                self._project_onto_simplex((p + alpha * losses).numpy())
            )

            # Evaluate half-step
            losses_h = []
            grad_h = torch.zeros_like(beta)
            for i, e in enumerate(E_unique):
                idx = env_t == e
                N_e = N_full_t[idx]
                y_e = y_t[idx]
                w_e = w_t[idx]
                pred_h = N_e @ beta_half
                residual_h = pred_h - y_e
                loss_h = torch.mean(w_e * residual_h**2)
                losses_h.append(loss_h)
                grad_e = (2.0 / idx.sum()) * (N_e.T @ (w_e * residual_h))
                grad_h += p_half[i] * grad_e
            losses_h = torch.stack(losses_h)
            grad_h += 2 * self.lam * (Omega_t @ beta_half)

            # Extragradient step 2: full step using half-step gradients
            beta = beta - alpha * grad_h
            p = torch.tensor(
                self._project_onto_simplex((p + alpha * losses_h).numpy())
            )

            # Evaluate at full step
            losses_new = []
            for i, e in enumerate(E_unique):
                idx = env_t == e
                N_e = N_full_t[idx]
                y_e = y_t[idx]
                w_e = w_t[idx]
                pred_new = N_e @ beta
                residual_new = pred_new - y_e
                loss_new = torch.mean(w_e * residual_new**2)
                losses_new.append(loss_new)
            losses_new = torch.stack(losses_new)

            max_loss = torch.max(losses_new)

            if verbose and t % (epochs // 10) == 0:
                obj = (p * losses_new).sum() + self.lam * (
                    beta @ (Omega_t @ beta)
                )
                print(f"Iter {t}: obj = {obj.item():.5f}")

            if best_max_loss - max_loss.item() > min_delta:
                best_max_loss = max_loss.item()
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if early_stopping and (epochs_no_improvement >= patience):
                if verbose:
                    print(
                        f"Early stopping at epoch {t}, best max_loss = {best_max_loss:.6f}"
                    )
                break

        return beta.detach().numpy(), p.detach().numpy()

    # TODO: maybe instead of doing k-fold cv we could simply split in train and validation
    #  in order to try out more lambda values.
    def _kfcv_cp(self, Omega: np.ndarray):
        lambdas = np.logspace(-3, 0, 15)
        K = 5
        cv_scores = np.zeros(len(lambdas))

        envs = np.unique(self.envbar)
        env_indices = {e: np.where(self.envbar == e)[0] for e in envs}

        # make K stratified folds to keep the proportions of environments
        np.random.seed(123)
        folds = [[] for _ in range(K)]
        for indices in env_indices.values():
            np.random.shuffle(indices)
            for i, idx in enumerate(np.array_split(indices, K)):
                folds[i].extend(idx)
        folds = [np.array(sorted(f)) for f in folds]

        for k in range(K):
            val_idx = folds[k]
            train_idx = np.setdiff1d(np.arange(len(self.ux)), val_idx)

            # training data
            ux_train = self.ux[train_idx]
            ybar_train = self.ybar[train_idx]
            wbar_train = self.wbar[train_idx]
            envbar_train = self.envbar[train_idx]
            N_train = self._bspline_dm(ux_train)

            # variable and constraints definition
            beta = cp.Variable(self.M)
            t = cp.Variable(nonneg=True)
            constraints = []
            for e in np.unique(envbar_train):
                mask = envbar_train == e
                Ne = N_train[mask]
                We = np.diag(wbar_train[mask])
                res = ybar_train[mask] - Ne @ beta
                constraints.append(cp.quad_form(res, We) / np.sum(mask) <= t)

            # validation data
            ux_val = self.ux[val_idx]
            ybar_val = self.ybar[val_idx]
            envbar_val = self.envbar[val_idx]
            N_val = self._bspline_dm(ux_val)

            beta_prev = None  # warm start

            for i, lam in enumerate(lambdas):
                objective = cp.Minimize(t + lam * cp.quad_form(beta, Omega))
                problem = cp.Problem(objective, constraints)

                if beta_prev is not None:
                    beta.value = beta_prev

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        problem.solve(warm_start=True, verbose=False)
                except Exception:
                    cv_scores[i] += np.inf
                    continue

                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    cv_scores[i] += np.inf
                    continue

                beta_prev = beta.value

                y_pred = N_val @ beta.value
                mse_envs = []
                for e in np.unique(envbar_val):
                    mask = envbar_val == e
                    mse_e = np.mean((ybar_val[mask] - y_pred[mask]) ** 2)
                    mse_envs.append(mse_e)
                cv_scores[i] += max(mse_envs)

        cv_scores /= K
        best_idx = np.argmin(cv_scores)
        self.lam = lambdas[best_idx]

    def _fit(self, **kwargs):
        # Use fine grid like in working code - at least 400 points
        n_grid_points = max(400, 10 * self.M)
        x_grid = np.linspace(self.x_min, self.x_max, n_grid_points)
        # x_grid = np.linspace(self.xbar.min(), self.xbar.max(), len(self.xbar))
        Omega = self._omega(x_grid)

        # Add small regularization to prevent singularity
        Omega += 1e-12 * np.eye(self.M)
        self.Omega = Omega

        if self.method == "erm":
            N = self._bspline_dm(self.ux)
            NTW = N.T * self.wbar
            NTWN = NTW @ N
            NTWy = NTW @ self.ybar
            if self.cv:
                lambdas = np.logspace(-3, 2, 30)
                cv_errors = []
                for lam in lambdas:
                    A = NTWN + lam * Omega
                    B = np.linalg.solve(A, NTW)
                    S = N @ B
                    yhat = S @ self.ybar
                    leverages = np.diag(S)
                    numer = (self.ybar - yhat) ** 2
                    denom = (1 - leverages) ** 2
                    loocv_error = np.mean(numer / denom)
                    cv_errors.append(loocv_error)
                best_idx = np.argmin(cv_errors)
                best_lambda = lambdas[best_idx]
                self.lam = best_lambda
            self.coef = np.linalg.solve(NTWN + self.lam * Omega, NTWy)
        else:
            # TODO: we may want to propose a thinning strategy for the matrix of second derivatives:
            #  this would allow for better stability and convergence with the optimization methods.
            if self.opt_method == "cp":
                if self.cv:
                    self._kfcv_cp(Omega)
                beta = cp.Variable(self.M)
                t = cp.Variable(nonneg=True)
                constraints = []
                for e in np.unique(self.envbar):
                    mask = self.envbar == e
                    count_env = np.sum(mask)
                    # Get the unique x values for this environment
                    x_env = self.ux[mask]
                    y_env = self.ybar[mask]
                    N_e = self._bspline_dm(x_env)
                    W_e = np.diag(self.wbar[mask])
                    residual = y_env - N_e @ beta
                    constraints.append(
                        cp.quad_form(residual, W_e) / count_env <= t
                    )
                    # constraints.append(cp.mean(cp.square(y_env - N_e @ beta)) <= t)

                objective = cp.Minimize(
                    t + self.lam * cp.quad_form(beta, Omega)
                )
                problem = cp.Problem(objective, constraints)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    problem.solve()

                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"Warning: Problem status is {problem.status}")

                self.coef = beta.value
            else:
                beta, _ = self._train_extragradient(**kwargs)
                self.coef = beta

        self.spline = BSpline(
            self.knots, self.coef, self.degree, extrapolate=False
        )

    def predict(self, x_new: np.ndarray):
        x_new = np.asarray(x_new).flatten()
        # x_scaled = (x_new - self.ux[0]) / self.r_ux
        # return self.spline(x_scaled)
        return self.spline(x_new)
