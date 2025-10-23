import numpy as np
from sklearn.ensemble import RandomForestRegressor
from adaXT.random_forest import RandomForest
from scipy.optimize import minimize
import cvxpy as cp


# =======
# MAGGING
# =======
class MaggingRF:
    """
    Distribution Generalization with Random Forest Regressor.
    Implements the magging estimator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        max_depth: int | None = None,
        max_features: int | str | None | float = 1.0,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        backend: str = "adaXT",
        risk: str = "nrw",
        sols_erm: np.ndarray | None = None,
        solver: str | None = None,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            For default parameters, see documentation of `sklearn.ensemble.RandomForestRegressor`.
            backend : Backend to use for fitting the forests.
                Possible values are {"sklearn", "adaXT"}
            risk : Risk definition (default "nrw")
                Possible values are {"mse", "nrw", "reg"}
            sols_erm : Solutions with ERM (default None)
                Must be provided if risk is "reg"
            solver : Solver to use (default None)
        """
        if backend not in ["sklearn", "adaXT"]:
            raise ValueError("backend must be one of 'sklearn', 'adaXT'.")
        if risk not in ["mse", "nrw", "reg"]:
            raise ValueError("risk must be one of 'mse', 'nrw', 'reg'.")
        if sols_erm is None and risk == "reg":
            raise ValueError("sols_erm must be provided if risk is 'reg'.")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.weights_magging = None
        self.backend = backend
        self.risk = risk
        self.sols_erm = sols_erm
        self.solver = solver
        self.model_list = []

    def get_weights(self) -> np.ndarray | None:
        return self.weights_magging

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        E_train: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the predictions for a given test dataset
        promoting those trees in the Random Forest that
        minimize the OOB prediction error.

        Args:
            X_train: Feature matrix of the training data.
            Y_train: Response vector of the training data.
            E_train: Environment label of the training data.

        Returns:
            wfitted: Weighted fitted values.
        """
        n_envs = len(np.unique(E_train))

        fitted_envs = []
        for env in np.unique(E_train):
            Xtr_e = X_train[E_train == env]
            Ytr_e = Y_train[E_train == env]
            if self.backend == "sklearn":
                rfm = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state,
                )
                rfm.fit(Xtr_e, Ytr_e)
            else:
                if self.max_depth is None:
                    self.max_depth = 2**31 - 1
                rfm = RandomForest(
                    forest_type="Regression",
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    seed=self.random_state,
                )
                rfm.fit(Xtr_e, Ytr_e)
            self.model_list.append(rfm)
            fitted_envs.append(rfm.predict(X_train))
        fitted_envs = np.column_stack(fitted_envs)

        if self.risk == "nrw":

            def objective(w: np.ndarray, F: np.ndarray) -> float:
                return np.dot(w.T, np.dot(F.T, F).dot(w))

            winit = np.array([1 / n_envs] * n_envs)
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = [[0, 1] for _ in range(n_envs)]
            wmag = minimize(
                objective,
                winit,
                args=(fitted_envs,),
                bounds=bounds,
                constraints=constraints,
            ).x
            self.weights_magging = wmag
        else:
            if self.risk == "mse":
                z = cp.Variable(nonneg=True)
            else:
                z = cp.Variable()
            wmag = cp.Variable(3, nonneg=True)
            constraints = []
            for e, env in enumerate(np.unique(E_train)):
                mask = E_train == env
                n_e = np.sum(mask)
                Xtr_e = X_train[mask]
                Ytr_e = Y_train[mask]
                fitted_models = []
                for k, env_k in enumerate(np.unique(E_train)):
                    fitted_models.append(self.model_list[k].predict(Xtr_e))
                fitted_models = np.column_stack(fitted_models)
                left = cp.sum_squares(Ytr_e - fitted_models @ wmag)
                if self.risk == "reg":
                    sols_erm_e = self.sols_erm[mask]
                    left -= cp.sum_squares(Ytr_e - sols_erm_e)
                constraints.append(left / n_e <= z)
            constraints.append(cp.sum(wmag) == 1)

            objective = cp.Minimize(z)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self.solver)

            wmag = wmag.value
            self.weights_magging = wmag

        wfitted = np.dot(wmag, fitted_envs.T)

        return wfitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts using the optimal weights found with magging

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of the test data.

        Returns
        -------
        preds : np.ndarray
            Predicted values.
        """
        preds_envs = []
        for model in self.model_list:
            preds_envs.append(model.predict(X))
        preds_envs = np.column_stack(preds_envs)
        preds = np.dot(self.weights_magging, preds_envs.T)
        return preds
