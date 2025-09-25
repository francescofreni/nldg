# code modified from https://github.com/zywang0701/DRoL/blob/main/methods/drol.py
from nldg.additional.utils import *
from nldg.additional.data import *
import cvxpy as cp
from adaXT.random_forest import RandomForest


class DRoL:
    def __init__(self, data, params, method="mse", sigma2=None, seed=42):
        self.data = data
        self.Gamma_plug = None
        self.Gamma_corr = None
        self.pred_full_mat = None
        self.source_full_models = None
        self.params = params
        self.method = method
        if self.method == "mse" and sigma2 is None:
            raise ValueError("sigma2 cannot be None when method is 'mse'.")
        self.sigma2 = sigma2
        self.seed = seed

    def fit(self, outcome_learner="xgb", density_learner="xgb"):
        """
        Compute the plug-in and bias-corrected estimators of the Gamma matrix.

        Args:
            outcome_learner (str, optional): method used to fit outcome models on each source. Defaults to 'xgb'.
            density_learner (str, optional): method used to fit density models on each source. Defaults to 'xgb'.
        """
        L = self.data.L
        nl_s = [self.data.X_sources_list[l].shape[0] for l in range(L)]
        N = self.data.X_target.shape[0]

        # ------------ Plug-in Estimator of Gamma matrix ------------
        # Use the entire source data to fit the source models
        # and predict on the target data
        self.pred_full_mat = np.zeros((N, L))
        self.source_full_models = [
            RandomForest(**self.params) for l in range(L)
        ]
        # self.source_full_models = [OutcomeModel(learner=outcome_learner, params=None) for l in range(L)]
        for l in range(L):
            self.source_full_models[l].fit(
                self.data.X_sources_list[l], self.data.Y_sources_list[l]
            )
            self.pred_full_mat[:, l] = self.source_full_models[l].predict(
                self.data.X_target
            )
        # The plug-in estimator of Gamma matrix
        self.Gamma_plug = self.pred_full_mat.T @ self.pred_full_mat / N

        # Use the sample-split source data to fit
        source_A_models = [RandomForest(**self.params) for l in range(L)]
        source_B_models = [RandomForest(**self.params) for l in range(L)]
        # source_A_models = [OutcomeModel(learner=outcome_learner, params=None) for l in range(L)]
        # source_B_models = [OutcomeModel(learner=outcome_learner, params=None) for l in range(L)]
        density_A_models = [
            DensityModel(learner=density_learner, params=None)
            for l in range(L)
        ]
        density_B_models = [
            DensityModel(learner=density_learner, params=None)
            for l in range(L)
        ]
        for l in range(L):
            half_l = nl_s[l] // 2
            source_A_models[l].fit(
                self.data.X_sources_list[l][:half_l],
                self.data.Y_sources_list[l][:half_l],
            )
            source_B_models[l].fit(
                self.data.X_sources_list[l][half_l:],
                self.data.Y_sources_list[l][half_l:],
            )
            density_A_models[l].fit(
                self.data.X_sources_list[l][:half_l],
                self.data.X_target,
                seed=self.seed,
            )
            density_B_models[l].fit(
                self.data.X_sources_list[l][half_l:],
                self.data.X_target,
                seed=self.seed,
            )

        # ------------ Bias-Corrected Estimator of Gamma matrix ------------
        self.Gamma_corr = self.Gamma_plug.copy()

        for k in range(L):
            fkA = source_A_models[k]
            fkB = source_B_models[k]
            wkA = density_A_models[k]
            wkB = density_B_models[k]
            half_k = nl_s[k] // 2

            for l in range(L):
                flA = source_A_models[l]
                flB = source_B_models[l]
                wlA = density_A_models[l]
                wlB = density_B_models[l]
                half_l = nl_s[l] // 2

                num1A = self._bias_correct(
                    fkA,
                    flA,
                    wlA,
                    self.data.X_sources_list[l][half_l:],
                    self.data.Y_sources_list[l][half_l:],
                )
                num2A = self._bias_correct(
                    flA,
                    fkA,
                    wkA,
                    self.data.X_sources_list[k][half_k:],
                    self.data.Y_sources_list[k][half_k:],
                )
                num1B = self._bias_correct(
                    fkB,
                    flB,
                    wlB,
                    self.data.X_sources_list[l][:half_l],
                    self.data.Y_sources_list[l][:half_l],
                )
                num2B = self._bias_correct(
                    flB,
                    fkB,
                    wkB,
                    self.data.X_sources_list[k][:half_k],
                    self.data.Y_sources_list[k][:half_k],
                )
                self.Gamma_corr[k, l] -= (num1A + num2A + num1B + num2B) / 2

        self.Gamma_corr = (self.Gamma_corr.T + self.Gamma_corr) / 2

    def predict(self, bias_correct=True, priors=None):
        """
        Estimate the optimal aggregation weights using the estimated Gamma matrix,
        and yield the robust prediction on the target domain.

        Args:
            bias_correct (bool, optional): whether to use the bias-corrected estimator. Defaults to True.
            priors (list, optional): priors upon the aggregation weights. Defaults to None.

        Returns:
            pred : the robust prediction on the target domain
            q_opt : the optimal aggregation weights
        """
        Gamma = self.Gamma_corr if bias_correct else self.Gamma_plug
        q = cp.Variable(self.data.L, nonneg=True)
        if priors is None:
            constraints = [cp.sum(q) == 1]
        else:
            prior_weight, rho = priors
            constraints = [cp.sum(q) == 1, cp.norm(q - prior_weight) <= rho]
        if self.method == "reward":
            objective = cp.Minimize(cp.quad_form(q, Gamma))
        elif self.method == "mse":
            gamma = np.diag(Gamma)
            linear = gamma + self.sigma2
            objective = cp.Minimize(
                cp.quad_form(q, Gamma) - cp.sum(cp.multiply(linear, q))
            )
        else:
            gamma = np.diag(Gamma)
            objective = cp.Minimize(
                cp.quad_form(q, Gamma) - cp.sum(cp.multiply(gamma, q))
            )
        prob = cp.Problem(objective, constraints)
        prob.solve()
        q_opt = q.value
        pred = self.pred_full_mat @ q_opt
        return pred, q_opt

    def _bias_correct(self, fk, fl, wl, Xl, Yl):
        """
        Compute the bias corrected term: mean[wl * fk(Xl) * (fl(Xl) - Yl)],
        where the models fk, fl, wl are independent of the data Xl, Yl.

        Args:
            fk (Instance of OutComeModel): fitted outcome model on the k-th source domain
            fl (Instance of OutcomeModel): fitted outcome model on the l-th source domain
            wl (Instance of DensityModel): fitted density ratio model on the l-th source domain
            Xl : feature matrix on the l-th source domain
            Yl : lable array on the l-th source domain
        """
        # preds = wl.predict(Xl)
        # print(np.min(preds), np.max(preds))
        return np.mean(wl.predict(Xl) * fk.predict(Xl) * (fl.predict(Xl) - Yl))
