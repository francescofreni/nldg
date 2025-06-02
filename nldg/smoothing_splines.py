import numpy as np
from scipy.interpolate import CubicSpline
import cvxpy as cp
import torch
import torch.optim as optim
import torch.nn.functional as F


def erm_ss(X, Y, lam):
    X = np.atleast_1d(X).flatten()

    idx = np.argsort(X)
    X = np.array(X)[idx]
    Y = np.array(Y)[idx]
    n = len(X)

    h = np.diff(X)

    Q = np.zeros((n - 2, n))
    for i in range(n - 2):
        hi, hip1 = h[i], h[i + 1]
        Q[i, i] = 1.0 / hi
        Q[i, i + 1] = -1.0 / hi - 1.0 / hip1
        Q[i, i + 2] = 1.0 / hip1

    R = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        R[i, i] = (h[i] + h[i + 1]) / 3.0
    for i in range(n - 3):
        R[i, i + 1] = h[i + 1] / 6.0
        R[i + 1, i] = h[i + 1] / 6.0

    M = np.linalg.solve(R, Q)
    K = Q.T @ M  # = Q^T R^{-1} Q

    A = np.eye(n) + lam * K
    g_sorted = np.linalg.solve(A, Y)

    g_unsorted = np.empty_like(g_sorted)
    g_unsorted[idx] = g_sorted

    cs = CubicSpline(X, g_sorted, bc_type="natural")

    # g = cp.Variable(n)
    # rss = cp.sum_squares(Y - g)
    # penalty = lam * cp.sum_squares(K @ g)
    # objective = rss + penalty
    # prob = cp.Problem(cp.Minimize(objective))
    # prob.solve(solver=cp.SCS)

    # g_sorted = g.value
    # g_unsorted = np.empty_like(g_sorted)
    # g_unsorted[idx] = g_sorted

    # cs = CubicSpline(X, g.value, bc_type='natural')

    return g_unsorted, g_sorted, cs


def minmax_ss(X, Y, E, lam, method="cp", **kwargs):
    X = np.atleast_1d(X).flatten()
    idx = np.argsort(X)
    X, Y, E = X[idx], Y[idx], E[idx]
    n = len(X)

    h = np.diff(X)

    Q = np.zeros((n - 2, n))
    for i in range(n - 2):
        hi, hip1 = h[i], h[i + 1]
        Q[i, i] = 1.0 / hi
        Q[i, i + 1] = -1.0 / hi - 1.0 / hip1
        Q[i, i + 2] = 1.0 / hip1

    R = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        R[i, i] = (h[i] + h[i + 1]) / 3.0
    for i in range(n - 3):
        R[i, i + 1] = h[i + 1] / 6.0
        R[i + 1, i] = h[i + 1] / 6.0

    M = np.linalg.solve(R, Q)
    K = Q.T @ M

    if method == "cp":
        g = cp.Variable(n)
        t = cp.Variable(nonneg=True)

        constraints = []
        for env in np.unique(E):
            mask = E == env
            constraints.append(
                cp.sum_squares(Y[mask] - g[mask]) / np.sum(mask) <= t
            )

        smooth_pen = lam * cp.sum_squares(K @ g)
        obj = cp.Minimize(t + smooth_pen)

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)

        g_sorted = g.value
        g_unsorted = np.empty_like(g_sorted)
        g_unsorted[idx] = g_sorted
        cs = CubicSpline(X, g_sorted, bc_type="natural")
        return g_unsorted, g_sorted, cs
    elif method == "gdro":
        g_sorted, p = train_spline_GDRO(Y, E, K, lam=lam, **kwargs)
        g_unsorted = np.empty_like(g_sorted)
        g_unsorted[idx] = g_sorted
        cs = CubicSpline(X, g_sorted, bc_type="natural")
        return g_unsorted, g_sorted, cs, p
    else:
        g_sorted = train_spline_subgradient(Y, E, K, lam=lam, **kwargs)
        g_unsorted = np.empty_like(g_sorted)
        g_unsorted[idx] = g_sorted
        cs = CubicSpline(X, g_sorted, bc_type="natural")
        return g_unsorted, g_sorted, cs


def train_spline_GDRO(
    Y,
    E,
    K,
    lr_g=1e-3,
    eta=0.1,
    lam=0.01,
    epochs=500,
    weight_decay=0.0,
    early_stopping=True,
    patience=10,
    min_delta=1e-6,
    verbose=False,
    seed=0,
):
    """
    Solve min_beta max_env MSE_env(beta) + lambda * beta^T Omega beta
    via a mirror-ascent / gradient-descent loop.

    Args:
      Y: array-like of shape (n,), target values.
      E: array-like of shape (n,), environment labels (integers or hashable).
      K: (n,n) penalty matrix.
      lr_g: stepsize for g-descent.
      eta: mirror-ascent stepsize for environment-weights.
      lam: regularization coefficient.
      epochs: max number of epochs.
      early_stopping: whether to use early stopping.
      patience: epochs with no improvement before stopping.
      min_delta: minimum improvement to reset patience.
      verbose: print progress.
      seed: random seed for reproducibility.

    Returns:
      g: numpy array of shape (n,) final estimated function values on the sorted X grid.
      p: numpy array of length E unique environments final environment weights.
      cs: CubicSpline fitted to (X_sorted, g).
    """
    torch.manual_seed(seed)

    envs, inv = np.unique(E, return_inverse=True)
    num_env = len(envs)
    n = len(Y)

    Y_t = torch.tensor(Y, dtype=torch.float32)
    inv_t = torch.tensor(inv, dtype=torch.long)
    K_t = torch.tensor(K, dtype=torch.float32)

    g = torch.zeros(n, requires_grad=True)
    optimizer = optim.Adam([g], lr=lr_g, weight_decay=weight_decay)

    p = torch.ones(num_env, dtype=torch.float32) / num_env

    best_obj = float("inf")
    best_g = None
    best_p = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        losses = []
        # Compute losses for each environment
        for i, env in enumerate(envs):
            mask = inv_t == i
            if mask.sum() == 0:
                losses.append(torch.tensor(0.0))
            else:
                pred = g[mask]
                true = Y_t[mask]
                losses.append(F.mse_loss(pred, true, reduction="mean"))
        losses = torch.stack(losses)

        # Mirror ascent for p
        with torch.no_grad():
            p = p * torch.exp(eta * losses)
            p = p / p.sum()

        # Compute weighted loss and regularizer
        weighted_loss = torch.dot(p, losses)
        reg = lam * (g @ (K_t @ g))
        obj = weighted_loss + reg

        # Gradient descent step on g
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        current = obj.item()
        # Early stopping / track best
        if current + min_delta < best_obj:
            best_obj = current
            best_g = g.detach().cpu().numpy().copy()
            best_p = p.detach().cpu().numpy().copy()
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}: obj={current:.6f}, best={best_obj:.6f}")
        if early_stopping and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Fallback if no improvement ever
    if best_g is None:
        best_g = g.detach().cpu().numpy()
        best_p = p.detach().cpu().numpy()

    return best_g, best_p


def train_spline_subgradient(
    Y, E, K, lr=1e-2, lam=1e-3, epochs=500, verbose=False, seed=0
):
    torch.manual_seed(seed)

    envs = np.unique(E)
    num_env = len(envs)
    n = len(Y)

    g = np.zeros(n)

    for epoch in range(1, epochs + 1):
        # 1) compute per‐env MSE
        losses = np.zeros(num_env)
        for e in range(num_env):
            mask = E == e
            if np.sum(mask) == 0:
                losses[e] = 0.0
            else:
                pred = g[mask]
                true = Y[mask]
                losses[e] = np.mean((pred - true) ** 2)

        # 2) pick worst‐case environment
        e_star = int(losses.argmax().item())
        f_star = losses[e_star].item()

        # 3) compute subgradient of f_{e*}
        mask = E == e_star
        Y_e = Y[mask]
        g_e = g[mask]
        n_e = np.sum(mask)

        # ∇_β MSE = (2/n_e) N_eᵀ (N_e β − y_e)
        grad_data = (2.0 / n_e) * (g_e - Y_e)

        # ∇_β reg = 2 λ K g
        grad_reg = 2 * lam * (K @ g)

        gk = grad_data + grad_reg

        # 4) subgradient descent update
        g = g - lr * gk

        if verbose and (epoch % (epochs // 10 or 1) == 0):
            obj = f_star + lam * (g @ (K @ g)).item()
            print(
                f"iter {epoch:4d}  env*={e_star}  MSE*={f_star:.4e}  obj≈{obj:.4e}"
            )

    return g
