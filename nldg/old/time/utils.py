from __future__ import division
import numpy as np
from scipy.linalg import block_diag


def generate_nonlinear_data(
    n, p, m, block_sizes, c_coeffs, OM, rng, test=False
):
    """
    Generates nonlinear data with invariant and time-varying components.

    Parameters:
    - n: int, number of samples
    - p: int, number of features
    - m: int, number of covariate shifts
    - block_sizes: list, sizes of diagonal blocks
    - c_coeffs: list, indices of constant (invariant) coefficients
    - OM: orthonormal transformation matrix
    - rng: numpy random generator
    - test: bool, whether to generate test data

    Returns:
    - X: Feature matrix
    - Y: Response variable
    - Sigma_list: Covariance matrices across time
    """

    mu_x = np.zeros(p)
    X = np.zeros((n, p))
    Y = np.zeros((n,))
    eps = 0.5 * rng.normal(size=(n,))
    Sigma_list = np.zeros((n, p, p))

    ws = int(n / m)
    w_start = [j * ws for j in range(m)]
    v_coeffs = [k for k in list(range(p)) if k not in c_coeffs]

    def f_inv(X):
        # return np.sin(X[:, c_coeffs[0]]) + 0.5 * np.log(1 + np.abs(X[:, c_coeffs[1]])) + 2. * X[:, c_coeffs[2]] ** 2
        beta_0 = np.array([0.2] * len(c_coeffs))
        return X[:, c_coeffs] @ beta_0

    def f_res(X, t):
        # return np.tanh(X[:, v_coeffs[0]]) * t + X[:, v_coeffs[1]]/(1 + np.exp(X[:, v_coeffs[2]])) * np.sin((t / n) * np.pi)
        def gamma_0(j):
            return 1 - 1.5 * (t / n) * (np.sin((j + 1) * t / n + (j + 1)) ** 2)

        gamma = np.array([gamma_0(0), gamma_0(1), gamma_0(9)])
        return X[t, v_coeffs] @ gamma

    # def f_res_test(X, t):
    #    return np.tanh(t*n) * X[:, v_coeffs[2]] ** 4 - 2. * np.cos(X[:, v_coeffs[1]])/t

    def f_res_test(X):
        shift = np.array([-1, -1, -1])
        return X[:, v_coeffs] @ shift

    if test:
        rng_sigma = np.random.default_rng(m)
    else:
        rng_sigma = np.random.default_rng(0)

    for idx, w in enumerate(w_start):
        if idx == m - 1:
            ws = n - w_start[-1]

        A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
        Sigma = OM.T @ (A @ A.T + 0.0 * np.eye(p)) @ OM

        for i in range(ws):
            Sigma_list[w + i, :, :] = Sigma

        X[w : w + ws, :] = rng.multivariate_normal(
            mean=mu_x, cov=Sigma, size=ws
        )

        # Xproj = X @ OM.T

        # if test:
        #    Y[w:w + ws] = f_inv(Xproj[w:w + ws, :]) + f_res_test(Xproj[w:w + ws, :], w) + eps[w:w + ws]
        # else:
        #    Y[w:w + ws] = f_inv(Xproj[w:w + ws, :]) + f_res(Xproj[w:w + ws, :], w) + eps[w:w + ws]

        if test:
            # Xproj = X @ OM.T
            # Y[w:w + ws] = f_res_test(Xproj[w:w + ws, :])
            Y[w : w + ws] = f_res_test(X[w : w + ws, :])

    # Xproj = X @ OM.T
    if not test:
        for t in range(n):
            # Y[t] = f_res(Xproj, t)
            Y[t] = f_res(X, t)
    # Y += f_inv(Xproj) + eps
    Y += f_inv(X) + eps

    return X, Y, Sigma_list
