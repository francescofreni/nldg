import numpy as np
from sklearn.linear_model import LinearRegression
from cvxopt.solvers import qp
from cvxopt import matrix


# =======
# MAGGING
# =======
class MaggingLR:
    """
    Implements the maximin effect estimator with magging.
    """

    def __init__(self) -> None:
        """
        Initialize the class instance.
        """
        self.beta_maximin = None
        self.weights = None
        self.beta_envs = []

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            X_train: Feature matrix of the training data.
            Y_train: Response vector of the training data.
            E_train: Environment label of the training data.

        Returns:
            wfitted: Weighted fitted values.
        """
        X = np.asarray(X_train)
        y = np.asarray(Y_train).reshape(-1)
        E = np.asarray(E_train)
        n, p = X.shape

        unique_envs = np.unique(E)
        n_envs = len(unique_envs)

        for env in unique_envs:
            mask = E == env
            X_env, y_env = X[mask], y[mask]
            lr = LinearRegression()
            lr.fit(X_env, y_env)
            self.beta_envs.append(lr.coef_.ravel())

        beta_hat = np.column_stack(self.beta_envs)
        Sigma_hat = (X.T @ X) / n
        H = beta_hat.T @ Sigma_hat @ beta_hat
        H = matrix(H, (n_envs, n_envs))
        A = matrix(
            np.concatenate((np.ones((1, n_envs)), np.eye(n_envs)), axis=0),
            (n_envs + 1, n_envs),
        )
        B = np.zeros((n_envs + 1, 1))
        B[0, 0] = 1
        B = matrix(B)
        Q = matrix(np.zeros((n_envs, 1)))

        weights = qp(H, Q, -A, -B, options={"show_progress": False})
        w = np.array(weights["x"]).reshape(-1)

        self.beta_maximin = beta_hat @ w
        self.weights = weights
        wfitted = X @ self.beta_maximin
        return wfitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts using the optimal weights found with magging

        Args:
            X (np.ndarray): Feature matrix of the test data.

        Returns:
            preds (np.ndarray): Predicted values.
        """
        if self.beta_maximin is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X)
        return X @ self.beta_maximin
