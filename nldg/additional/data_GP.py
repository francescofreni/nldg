# code structure modified from https://github.com/zywang0701/DRoL/blob/main/methods/data.py

import numpy as np
from typing import Callable
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class DataContainer:
    def __init__(
        self,
        n: int,
        N: int,
        change_X_distr: bool = False,
        risk: str = "mse",
        d: int = 5,
        target_mode: str = "convex_mixture_P",
        common_core_func: bool = False,
    ) -> None:
        self.n = n  # number of samples per source environment
        self.N = N  # number of samples in the target domain
        self.d = d  # number of features
        self.n_E = None  # number of source environments

        # storage for generated data
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []

        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
        self.Q_target_list = []  # convex weights per test environment

        # list of source conditional outcome functions
        self.f_funcs = []

        # flags and risk
        self.change_X_distr = change_X_distr
        self.risk = risk
        self.target_mode = target_mode
        self.common_core_func = common_core_func

        # X distribution support
        self.X_supp = 1.0

        # noise stddev and GP hyperparams
        self.sigma_eps = 0.1
        self.gp_length_scale = 0.5
        self.gp_amplitude = 1.0
        self.n_grid = 1000  # grid size for GP function sampling
        # per-environment X distribution params (used when change_X_distr)
        # List of dicts with keys 'alpha', 'beta'
        self.env_x_params = None

    # -----------------------------
    # public API
    # -----------------------------
    def generate_funcs_list(self, n_E: int, seed: int | None = None) -> None:
        np.random.seed(seed)
        self.n_E = n_E
        base_funcs = [self._sample_additive_gp_function() for _ in range(n_E)]

        if self.common_core_func:
            self.f_funcs = [self._wrap_with_core(g_fn) for g_fn in base_funcs]
        else:
            self.f_funcs = base_funcs

    def generate_data(self, seed: int | None = None) -> None:
        np.random.seed(seed)
        self._reset_lists()

        if self.change_X_distr:
            # Initialize environment-specific X distribution parameters
            self._init_env_x_params(self.n_E)

        # ------- Generate Source Data -------
        for env_idx in range(self.n_E or 0):
            X = self._sample_X_source_env(env_idx, self.n)

            eps = np.random.normal(0.0, self.sigma_eps, size=self.n)
            Y = self.f_funcs[env_idx](X) + eps
            self.E_sources_list.append(np.full(self.n, env_idx, dtype=int))

            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)

        # ------- Generate Target Data (mode: convex_mixture_P or same)

        if self.target_mode == "same":
            eye_weights = np.eye(self.n_E)
            for env_idx in range(self.n_E):
                X_target = self._sample_X_source_env(env_idx, self.N)

                eps_t = np.random.normal(0.0, self.sigma_eps, size=self.N)
                Y_target = self.f_funcs[env_idx](X_target) + eps_t

                self.X_target_list.append(X_target)
                self.Y_target_potential_list.append(Y_target)
                self.E_target_potential_list.append(
                    np.full(self.N, env_idx, dtype=int)
                )
                self.Q_target_list.append(eye_weights[env_idx])

        elif self.target_mode == "convex_mixture_P":
            # For each test env, draw q, then sample (X,Y) via env draws
            Q = np.random.dirichlet(alpha=np.ones(self.n_E), size=self.n_E)
            for env_idx in range(self.n_E):
                q = Q[env_idx]
                # sample latent training env index per sample
                env_choices = np.random.choice(self.n_E, size=self.N, p=q)
                X_list = []
                Y_list = []
                for e in env_choices:
                    Xe = self._sample_X_source_env(int(e), 1)
                    ye = self.f_funcs[int(e)](Xe) + np.random.normal(
                        0.0, self.sigma_eps, size=1
                    )
                    X_list.append(Xe[0])
                    Y_list.append(ye[0])
                X_target_e = np.vstack(X_list)
                Y_target_e = np.array(Y_list)

                self.X_target_list.append(X_target_e)
                self.Y_target_potential_list.append(Y_target_e)
                self.E_target_potential_list.append(
                    np.full(self.N, env_idx, dtype=int)
                )
                self.Q_target_list.append(q)

    # -----------------------------
    # internal funcs
    # -----------------------------
    def _reset_lists(self) -> None:
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []
        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
        self.Q_target_list = []

    def _sample_X_source_env(self, env_idx: int, n: int) -> np.ndarray:
        """Sample X from the training environment-specific distribution.

        Extension point: override or modify to allow different X distributions
        per training environment in the future.
        """
        if self.change_X_distr and self.env_x_params is not None:
            params = self.env_x_params[int(env_idx)]
            alpha = params["alpha"]
            beta = params["beta"]
            # Draw per-feature Beta and map to [-X_supp, X_supp]
            U = np.random.beta(alpha, beta, size=(n, self.d))
            X = (2.0 * U - 1.0) * self.X_supp
            return X
        return self._sample_X_source(n)

    def _init_env_x_params(self, n_E: int) -> None:
        """Create per-environment Beta(alpha, beta) params for X if enabled.

        Alpha and beta are sampled uniformly from [1, 2] per environment and
        applied to all features independently. Values are stored so that
        convex_mixture_P can resample from the same marginals later.
        """
        self.env_x_params = []
        for _ in range(n_E):
            alpha = np.random.uniform(1.0, 2.0)
            beta = np.random.uniform(1.0, 2.0)
            self.env_x_params.append({"alpha": alpha, "beta": beta})

    def _sample_X_source(self, n: int) -> np.ndarray:
        # Uniform on [-X_supp, X_supp]^d
        low = -self.X_supp
        high = self.X_supp
        return np.random.uniform(low, high, size=(n, self.d))

    def _core_func(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return 1.5 * np.sum(np.abs(X), axis=1)

    def _wrap_with_core(
        self, g_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def f_wrapped(X: np.ndarray) -> np.ndarray:
            return self._core_func(X) + g_fn(X)

        return f_wrapped

    # -----------------------------
    # GP helpers
    # -----------------------------
    def _rbf_kernel(self, x_grid: np.ndarray) -> np.ndarray:
        kernel = ConstantKernel(self.gp_amplitude) * RBF(
            length_scale=self.gp_length_scale
        )
        K = kernel(x_grid, x_grid)

        return K

    def _sample_gp_function_1d(self) -> Callable[[np.ndarray], np.ndarray]:
        """Draw a 1D GP function on a grid and return an interpolator."""

        # grid for kernel computation
        x_grid = np.linspace(-self.X_supp, self.X_supp, self.n_grid).reshape(
            -1, 1
        )

        K = self._rbf_kernel(x_grid)

        f_vals = np.random.multivariate_normal(
            mean=np.zeros(self.n_grid), cov=K
        )

        def g(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x).reshape(-1)
            return np.interp(x.ravel(), x_grid.ravel(), f_vals)

        return g

    def _sample_additive_gp_function(
        self,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """f(X) = sum_j beta_j * g_j(X[:, j])
        with independent GP components g_j.
        """
        g_list = [self._sample_gp_function_1d() for _ in range(self.d)]

        beta = np.random.uniform(0.0, 1.0, size=(self.d,))

        def f_nd(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[None, :]
            out = np.zeros(X.shape[0])
            for j in range(self.d):
                out += beta[j] * g_list[j](X[:, j])
            return out

        f_nd.beta = beta
        return f_nd
