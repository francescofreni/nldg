# code modified from https://github.com/zywang0701/DRoL/blob/main/methods/data.py
import numpy as np


class DataContainer:
    def __init__(
        self, n, N, change_X_distr=False, risk="mse", unbalanced_envs=False
    ):
        self.n = n  # number of samples in each source domain
        self.N = N  # number of samples in the target domain
        self.d = 5  # number of features
        self.L = None  # number of source domains

        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.E_sources_list = []  # list of source environment labels

        self.X_target = None  # target covariate matrix
        self.X_target_list = (
            []
        )  # list of repeated target covariate matrix (just to make predictions)
        self.Y_target_potential_list = (
            []
        )  # list of potential target outcome vectors
        self.E_target_potential_list = (
            []
        )  # list of potential target environment labels

        self.f_funcs = []  # list of source conditional outcome functions
        self.mu0 = None  # target covariate distribution mean, used when generating data
        self.mu = None  # source covariate distribution mean
        self.Sigma0 = None  # target covariate distribution covariance, used when generating data
        self.Sigma = None  # source covariate distribution covariance
        self.change_X_distr = (
            change_X_distr  # whether the target X marginal is different or not
        )
        self.risk = risk  # Risk definition
        self.unbalanced_envs = unbalanced_envs

    def load_custom_data(self, Xtr, Ytr, Etr, Xte, Yte, Ete):
        """
        Load custom data into the DataContainer format.
        
        Parameters:
        -----------
        Xtr : numpy.ndarray
            Training features, shape (n_train, d)
        Ytr : numpy.ndarray
            Training outcomes, shape (n_train,)
        Etr : numpy.ndarray
            Training environment labels, shape (n_train,)
        Xte : numpy.ndarray
            Test features, shape (n_test, d)
        Yte : numpy.ndarray
            Test outcomes, shape (n_test,)
        Ete : numpy.ndarray
            Test environment labels, shape (n_test,)
        """
        self._reset_lists()

        # ensure if input are numpy arrays
        Xtr = np.array(Xtr)
        Ytr = np.array(Ytr)
        Etr = np.array(Etr)
        Xte = np.array(Xte)
        Yte = np.array(Yte)
        Ete = np.array(Ete)
        
        # Update dimensions based on data
        self.d = Xtr.shape[1]
        self.N = Xte.shape[0]
        
        # Get unique environments
        unique_envs = np.unique(Etr)
        self.L = len(unique_envs)
        
        # Split source data by environment
        for env in unique_envs:
            mask = Etr == env
            self.X_sources_list.append(Xtr[mask])
            self.Y_sources_list.append(Ytr[mask])
            self.E_sources_list.append(Etr[mask])
        
        # Set target data
        self.X_target = Xte
        
        # Split target data by environment for potential outcomes
        unique_target_envs = np.unique(Ete)
        for env in unique_target_envs:
            mask = Ete == env
            self.X_target_list.append(Xte[mask])
            self.Y_target_potential_list.append(Yte[mask])
            self.E_target_potential_list.append(Ete[mask])
        
        # Compute statistics
        self.mu = np.mean(Xtr, axis=0)
        self.Sigma = np.cov(Xtr.T)
        self.mu0 = np.mean(Xte, axis=0)
        self.Sigma0 = np.cov(Xte.T)
        
        # Update n based on average samples per environment
        self.n = int(np.mean([len(X) for X in self.X_sources_list]))

    def generate_funcs_list(self, L, seed=None):
        np.random.seed(seed)
        self.L = L
        beta_list = []
        A_list = []
        if not self.change_X_distr:
            self.mu0 = np.zeros(self.d)
        else:
            self.mu0 = np.array([-0.25, -0.25, 0, 0.25, 0.25])
            # self.mu0 = np.array([-0.25, -0.25, 0.5, 0.25, 0.25])
        for l in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)
            beta_list.append(beta)

            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2
            A_list.append(A)

            # compute c = trace(A) + mu^T A mu
            c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))

            def f_func(x, beta=beta, A=A, c=c):
                return (
                    np.sin(x.dot(beta)) + np.sum(np.dot(x, A) * x, axis=1) - c
                )

            self.f_funcs.append(f_func)

    def generate_data(self, seed=None):
        np.random.seed(seed)
        self._reset_lists()

        # ------- Generate Source Data -------
        mu = np.zeros(self.d)
        self.mu = mu
        Sigma = np.eye(self.d)
        self.Sigma = Sigma
        for l in range(self.L):
            if not self.unbalanced_envs:
                X = np.random.multivariate_normal(mu, Sigma, self.n)
                Y = self.f_funcs[l](X) + np.random.randn(self.n)
                self.E_sources_list.append(np.full(self.n, l, dtype=int))
            else:
                if l == 0:
                    X = np.random.multivariate_normal(mu, Sigma, self.n * 3)
                    Y = self.f_funcs[l](X) + np.random.randn(self.n * 3)
                    self.E_sources_list.append(
                        np.full(self.n * 3, l, dtype=int)
                    )
                else:
                    X = np.random.multivariate_normal(mu, Sigma, self.n)
                    Y = self.f_funcs[l](X) + np.random.randn(self.n)
                    self.E_sources_list.append(np.full(self.n, l, dtype=int))
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)

        # ------- Generate Target Data -------
        self.Sigma0 = np.eye(self.d)
        self.X_target = np.random.multivariate_normal(
            self.mu0, self.Sigma0, self.N
        )
        for l in range(self.L):
            Y_target = self.f_funcs[l](self.X_target) + np.random.randn(self.N)
            self.X_target_list.append(self.X_target)
            self.Y_target_potential_list.append(Y_target)
            self.E_target_potential_list.append(np.full(self.N, l, dtype=int))

    def _reset_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []
        self.X_target_list = []
        self.Y_target_potential_list = []
        self.E_target_potential_list = []
