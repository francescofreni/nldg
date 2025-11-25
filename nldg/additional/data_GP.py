import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class DataContainer:
    def __init__(
        self,
        n: int,
        N: int,
        L: int,
        d: int,
        change_X_distr: bool = False,
        risk: str = "mse",
        beta_low: float = 1.0,
        beta_high: float = 2.0,
    ) -> None:
        self.rng = np.random.default_rng()

        self.n = n  # number of samples per source environment
        self.N = N  # number of samples in the target domain
        self.d = d  # number of features
        self.L = L  # number of source environments

        # storage for generated data
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []
        self.f_sources_list = []

        self.X_target = None
        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
        self.f_target_potential_list = []

        # flags and risk
        self.change_X_distr = change_X_distr
        self.beta_low = beta_low
        self.beta_high = beta_high

        self.risk = risk

        # X distribution support
        self.X_supp = 1.0

        # List of dicts with keys 'alpha', 'beta'
        # per-environment X distribution params (used when change_X_distr)
        self.env_x_params = []

        # noise stddev and GP hyperparams
        self.sigma_eps = 0.25
        self.gp_length_scale = 0.5
        self.gp_variance = 1.0
        self.t = 1e-7  # jitter for numerical stability

    # -----------------------------
    # public API
    # -----------------------------

    def generate_dataset(
        self, seed: int | None = None, reuse_params: bool = False
    ) -> None:
        """
        Generate source and target Gaussian process datasets for all environments.

        Parameters
        ----------
        seed:
            Optional random seed used to re-initialize the internal RNG; if
            ``None``, the existing RNG state is used.
        reuse_params:
            If True, reuse previously stored environment-specific X
            distribution parameters in env_x_params; if False, draw new
            parameters. The shared target covariates X_target are resampled
            on every call when change_X_distr is False.
        """

        self.rng = np.random.default_rng(seed)
        self._reset_lists()

        if not reuse_params:
            self.env_x_params = []

        if self.change_X_distr and len(self.env_x_params) == 0:
            # initialize environment-specific X distribution parameters
            self._init_env_x_params()

        if not self.change_X_distr:
            self.X_target = self.sample_X_env(env_idx=0, n=self.N)

        for env_idx in range(self.L):
            # sample covariates for train/test
            X_tr = self.sample_X_env(env_idx, self.n)

            if self.change_X_distr:
                X_te = self.sample_X_env(env_idx, self.N)
            else:
                X_te = self.X_target

            # sample GP functions and outcomes for train/test
            f_tr = self._sample_gp_new(X_tr)
            f_te = self._sample_gp_conditional(
                X_obs=X_tr, f_obs=f_tr, X_new=X_te
            )

            # add noise to outcomes
            # (one could also add self.sigma_eps^2 I to the kernel)
            y_tr = f_tr + self.rng.normal(0.0, self.sigma_eps, size=self.n)
            y_te = f_te + self.rng.normal(0.0, self.sigma_eps, size=self.N)

            # store generated data
            self.X_sources_list.append(X_tr)
            self.Y_sources_list.append(y_tr)
            self.E_sources_list.append(np.full(self.n, env_idx, dtype=int))
            self.f_sources_list.append(f_tr)

            self.X_target_list.append(X_te)
            self.Y_target_potential_list.append(y_te)
            self.E_target_potential_list.append(
                np.full(self.N, env_idx, dtype=int)
            )
            self.f_target_potential_list.append(f_te)

    def _reset_lists(self) -> None:
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []
        self.f_sources_list = []
        self.X_target = None
        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
        self.f_target_potential_list = []

    # -----------------------------
    # Public evaluation helpers
    # -----------------------------

    def env_posterior_mean(
        self, env_idx: int, X_new: np.ndarray
    ) -> np.ndarray:
        """Evaluate the posterior mean of environment env_idx at X_new.

        Uses the GP prior and the stored training function values (X_tr, f_tr)
        for that environment to compute E[f(X_new) | f(X_tr) = f_tr].

        Parameters
        - env_idx: int, environment index (0..L-1)
        - X_new: (m, d) array of input locations

        Returns
        - mu: (m,) posterior mean at X_new
        """
        X_obs = self.X_sources_list[env_idx]
        f_obs = self.f_sources_list[env_idx]

        K_xx = self._kernel(X_obs, X_obs)
        K_xs = self._kernel(X_obs, X_new)
        K_sx = K_xs.T

        L = np.linalg.cholesky(K_xx + self.t * np.eye(K_xx.shape[0]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_obs))
        mu = K_sx @ alpha
        return mu.ravel()

    def sample_X_env(self, env_idx: int, n: int) -> np.ndarray:
        """Sample X from the training environment-specific distribution."""
        if self.change_X_distr and self.env_x_params is not None:
            params = self.env_x_params[int(env_idx)]
            alpha = params["alpha"]
            beta = params["beta"]
            # Draw per-feature Beta and map to [-X_supp, X_supp]
            U = self.rng.beta(alpha, beta, size=(n, self.d))
            X = (2.0 * U - 1.0) * self.X_supp
            return X
        else:
            return self.rng.uniform(
                -self.X_supp, self.X_supp, size=(n, self.d)
            )

    def sample_additional_X(self, n: int) -> list[np.ndarray]:
        """Return additional covariates from all environments.

        Produces a list with one (n, d) array per environment.
        """
        covariates_list = []
        for env_idx in range(self.L):
            X_env = self.sample_X_env(env_idx, n)
            covariates_list.append(X_env)
        return covariates_list

    def _init_env_x_params(self) -> None:
        """Create per-environment Beta(alpha, beta) params for X if enabled.

        Alpha and beta are sampled uniformly from [beta_low, beta_high] per
        environment and applied to all features independently.
        """
        self.env_x_params = []
        for _ in range(self.L):
            alpha = self.rng.uniform(self.beta_low, self.beta_high)
            beta = self.rng.uniform(self.beta_low, self.beta_high)
            self.env_x_params.append({"alpha": alpha, "beta": beta})

    # -----------------------------
    # GP helpers
    # -----------------------------

    def _kernel(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Kernel matrix between X and Z."""
        # X: (n,d), Z: (m,d)
        X = np.atleast_2d(X)
        Z = np.atleast_2d(Z)

        kernel = ConstantKernel(self.gp_variance) * RBF(
            length_scale=self.gp_length_scale
        )
        K = kernel(X, Z)
        return K

    def _sample_gp_new(self, X: np.ndarray) -> np.ndarray:
        """Draw f ~ p(f(X)) from the GP prior."""
        K = self._kernel(X, X)
        # cholesky with small jitter for numerical stability
        L = np.linalg.cholesky(K + self.t * np.eye(K.shape[0]))
        z = self.rng.standard_normal(K.shape[0])
        return L @ z

    def _sample_gp_conditional(
        self, X_obs: np.ndarray, f_obs: np.ndarray, X_new: np.ndarray
    ) -> np.ndarray:
        """
        Draw f_new ~ p(f(X_new) | f(X_obs)) from the GP prior.
        """
        K_xx = self._kernel(X_obs, X_obs)
        K_xs = self._kernel(X_obs, X_new)
        K_sx = K_xs.T
        K_ss = self._kernel(X_new, X_new)

        # solve with Cholesky
        L = np.linalg.cholesky(K_xx + self.t * np.eye(K_xx.shape[0]))
        # α = K_xx^{-1} f_obs via solves
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_obs))
        mu = K_sx @ alpha

        # Compute posterior cov: K_ss - K_sx K_xx^{-1} K_xs
        v = np.linalg.solve(L, K_xs)
        Sigma = K_ss - v.T @ v

        # Draw one sample
        L_post = np.linalg.cholesky(Sigma + self.t * np.eye(Sigma.shape[0]))
        z = self.rng.standard_normal(Sigma.shape[0])
        f_new = mu + L_post @ z
        return f_new
