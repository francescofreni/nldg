import warnings

import numpy as np
import cvxpy as cp
import scipy.sparse as sps
from pygam import LinearGAM
from pygam.utils import check_X, check_y, check_X_y, check_array, check_lengths

EPS = np.finfo(np.float64).eps  # machine epsilon


class MaxRMLinearGAM(LinearGAM):
    """MaxRM Linear GAM.

    This is a modification of Linear GAM that minimizes the maximum risk across environments.

    For a complete documentation, see https://github.com/dswah/pyGAM/blob/main/pygam/pygam.py
    """

    def __init__(
        self,
        terms="auto",
        max_iter=100,
        tol=1e-4,
        scale=None,
        callbacks=["deviance", "diffs"],
        fit_intercept=True,
        verbose=False,
        **kwargs,
    ):
        super(MaxRMLinearGAM, self).__init__(
            terms=terms,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            callbacks=callbacks,
            fit_intercept=fit_intercept,
            verbose=verbose,
            **kwargs,
        )

    def fit(
        self, X, y, env, weights=None, risk="mse", sols_erm=None, solver="ECOS"
    ):
        """
        LinearGAM that minimizes the maximum risk across environments.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors
        y : array-like, shape (n_samples,)
            Target values
        env : array-like, shape (n_samples,)
            Environment labels
        weights : array-like shape (n_samples,) or None, optional
            Sample weights.
            if None, defaults to array of ones
        risk : string, optional (default="mse")
            Risk function (values in ["mse", "nrw", "reg"])
        sols_erm : array-like, shape (n_samples,)
            Fitted values with empirical risk minimization
        solver : str, optional (default="ECOS")
            Solver used by cvxpy. Examples are 'ECOS', 'SCS', 'CLARABEL'

        Returns
        -------
        self : object
            Returns fitted GAM object

        Usage
        -----
        gam = MaxRMLinearGAM(terms=...)
        gam.fit(X, y, env=env, weights=weights, solver='ECOS')
        yhat = gam.predict(Xnew)
        """
        if risk not in ["mse", "nrw", "reg"]:
            raise ValueError("risk must be one of ['mse', 'nrw', 'reg']")
        if risk == "reg" and sols_erm is None:
            raise ValueError("if risk is 'reg', sols_erm must be provided")

        self._validate_params()

        # validate data
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        X = check_X(X, verbose=self.verbose)
        check_X_y(X, y)
        if len(env) != len(y):
            raise ValueError(
                f"inconsistent environment and response shape. found env: {env.shape}, y: {y.shape}"
            )
        env = np.asarray(env).ravel()
        env = check_array(
            env,
            force_2d=False,
            min_samples=1,
            ndim=1,
            name="env data",
            verbose=False,
        )

        if weights is not None:
            weights = np.array(weights).astype("f").ravel()
            weights = check_array(
                weights, name="sample weights", ndim=1, verbose=self.verbose
            )
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype("float64")

        # validate data-dependent parameters
        self._validate_data_dep_params(X)

        # begin capturing statistics
        self.statistics_ = {}
        self.statistics_["n_samples"] = len(y)
        self.statistics_["m_features"] = X.shape[1]

        # build a basis matrix for the GLM
        modelmat = self._modelmat(X)
        n, m = modelmat.shape

        # smoothing penalty P and (optional) constraints C
        P = self._P()  # (m x m) sparse PSD, already multiplied by lambdas
        S = sps.diags(np.ones(m) * np.sqrt(EPS))
        R = P + S
        if self.terms.hasconstraint:
            C = self._C()
            R += C

        # variables
        beta = cp.Variable(m)
        if risk == "mse":
            t = cp.Variable(nonneg=True)
        else:
            t = cp.Variable()

        # constraints
        constraints = []
        unique_envs = np.unique(env)

        for e in unique_envs:
            mask = env == e
            Xe = modelmat[mask, :]
            ye = y[mask]
            we = weights[mask]
            sqrt_we = np.sqrt(we)
            we_sum = np.sum(we)
            pred_e = cp.matmul(Xe, beta)
            residual = cp.multiply(sqrt_we, ye - pred_e)
            left = cp.sum_squares(residual)
            if risk == "nrw":
                left -= cp.sum_squares(cp.multiply(sqrt_we, ye))
            elif risk == "reg":
                sols_erm_e = sols_erm[mask]
                left -= cp.sum_squares(cp.multiply(sqrt_we, ye - sols_erm_e))
            constraints.append(left / we_sum <= t)

        # objective
        if sps.issparse(R):
            R = R.toarray()
        R_psd = cp.psd_wrap(R)
        objective = cp.Minimize(t + cp.quad_form(beta, R_psd))

        # problem
        prob = cp.Problem(objective, constraints)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prob.solve(solver=solver)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MaxRM solve failed: status={prob.status}")

        # store
        self.coef_ = beta.value

        return self
