# code structure modified from https://github.com/zywang0701/DRoL/blob/main/methods/data.py
import numpy as np


class DataContainer:
    def __init__(
        self,
        n: int,
        N: int,
        change_X_distr: bool = False,
        risk: str = "mse",
        d: int = 5,
        target_mode: str = "convex_mixture_P",
    ):
        self.n = n  # number of samples in each source domain
        self.N = N  # number of samples in the target domain
        self.d = d  # number of features
        self.L = None  # number of source domains

        # storage for generated data
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []

        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
        self.Q_target_list = []  # convex weights per test environment

        self.f_funcs = []  # list of source conditional outcome functions

        # flags and risk
        self.change_X_distr = change_X_distr
        self.risk = risk
        self.target_mode = target_mode

        # X distribution support
        self.X_supp = 3.0

        # noise stddev and GP hyperparams
        self.sigma_eps = 0.1

        # per-environment X distribution params (used when change_X_distr)
        # List of dicts with keys 'alpha', 'beta'
        self.env_x_params = []

    def generate_funcs_list(self, L: int, seed: int | None = None) -> None:
        np.random.seed(seed)
        self.L = L
        self.f_funcs = []

        for _ in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)

            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2

            def f_func(
                x,
                beta=beta,
                A=A,
            ) -> np.ndarray:
                return (
                    np.sin(x.dot(beta))
                    + np.sum(np.dot(x, A) * x, axis=1)
                    - np.trace(A)
                )

            self.f_funcs.append(f_func)

    def generate_data(
        self, seed: int | None = None, reuse_params: bool = False
    ) -> None:
        np.random.seed(seed)
        self._reset_lists()

        if not reuse_params:
            self.Q_target_list = []
            self.env_x_params = []

        if self.change_X_distr and len(self.env_x_params) == 0:
            # Initialize environment-specific X distribution parameters
            self._init_env_x_params(self.L)

        # ------- Generate Source Data -------
        for env_idx in range(self.L):
            X = self._sample_X_source_env(env_idx, self.n)

            eps = np.random.normal(0.0, self.sigma_eps, size=self.n)
            Y = self.f_funcs[env_idx](X) + eps
            self.E_sources_list.append(np.full(self.n, env_idx, dtype=int))

            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)

        # ------- Generate Target Data (mode: convex_mixture_P or same)

        if self.target_mode == "same":
            eye_weights = np.eye(self.L)
            for env_idx in range(self.L):
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
            if reuse_params and len(self.Q_target_list) == self.L:
                Q = np.array(self.Q_target_list)
            else:
                Q = np.random.dirichlet(alpha=np.ones(self.L), size=self.L)

            self.Q_target_list = Q.tolist()

            for env_idx in range(self.L):
                q = Q[env_idx]
                # sample latent training env index per sample
                env_choices = np.random.choice(self.L, size=self.N, p=q)
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

    def _reset_lists(self) -> None:
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []
        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []

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

    def _init_env_x_params(self, L: int) -> None:
        """Create per-environment Beta(alpha, beta) params for X if enabled.

        Alpha and beta are sampled uniformly from [1, 2] per environment and
        applied to all features independently. Values are stored so that
        convex_mixture_P can resample from the same marginals later.
        """
        self.env_x_params = []
        for _ in range(L):
            alpha = np.random.uniform(1.0, 2.0)
            beta = np.random.uniform(1.0, 2.0)
            self.env_x_params.append({"alpha": alpha, "beta": beta})

    def _sample_X_source(self, n: int) -> np.ndarray:
        # Uniform on [-X_supp, X_supp]^d
        low = -self.X_supp
        high = self.X_supp
        return np.random.uniform(low, high, size=(n, self.d))
