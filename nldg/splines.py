import numpy as np
from typing import List, Tuple
import random
from scipy.interpolate import BSpline
from patsy import dmatrix
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def bspline_N(x, knots, degree):
    """
    Evaluate the B-spline basis at points x.
    """
    x = np.atleast_1d(x).flatten()
    M = len(knots) - degree - 1
    X_design = np.zeros((len(x), M))
    for i in range(M):
        c = np.zeros(M)
        c[i] = 1
        spl = BSpline(knots, c, degree, extrapolate=False)
        X_design[:, i] = spl(x)
    return X_design


def bspline_2der(x, knots, degree, i):
    """
    Evaluate the second derivative of the i-th B-spline basis function at x.
    """
    x = np.atleast_1d(x).flatten()
    M = len(knots) - degree - 1
    c = np.zeros(M)
    c[i] = 1
    spl = BSpline(knots, c, degree, extrapolate=False)
    spl2 = spl.derivative(nu=2)
    return spl2(x)


def omega(knots, degree, grid_points):
    """
    Compute Omega[i,j] = \int [B_i''(x) B_j''(x)] dx by numerical integration.
    """
    M = len(knots) - degree - 1
    Omega = np.zeros((M, M))
    dx = grid_points[1] - grid_points[0]  # assume uniform grid spacing
    for i in range(M):
        Bi2 = bspline_2der(grid_points, knots, degree, i)
        for j in range(M):
            Bj2 = bspline_2der(grid_points, knots, degree, j)
            Omega[i, j] = np.sum(Bi2 * Bj2) * dx
    return Omega


def plot_dtr(dtr, x_grid, preds_erm, preds_maximin, optfun=None):
    line_colors = ["lightskyblue", "orange"]
    data_colors = ["black", "grey", "silver"]
    environments = sorted(dtr["E"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, env in enumerate(environments):
        marker_style = "o"
        ax.scatter(
            dtr[dtr["E"] == env]["X"],
            dtr[dtr["E"] == env]["Y"],
            color=data_colors[idx],
            marker=marker_style,
            alpha=0.5,
            s=30,
            label=f"Env {env + 1}",
        )

    ax.plot(x_grid, preds_erm, color=line_colors[0], linewidth=2, label="SS")
    ax.plot(
        x_grid,
        preds_maximin,
        color=line_colors[1],
        linewidth=2,
        label="MaximinSS",
    )

    if optfun == 1:
        y_opt = 0.8 * np.sin(x_grid / 2) ** 2 + 3
        ax.plot(x_grid, y_opt, color="orangered", linewidth=3, label="Optimal")
    elif optfun == 2:
        y_opt = np.where(x_grid > 0, 2.4 * x_grid, -2.4 * x_grid)
        ax.plot(x_grid, y_opt, color="orangered", linewidth=3, label="Optimal")
    elif optfun == 3:
        y_opt = np.where(x_grid > 0, 1.86 * x_grid, 1.63 * x_grid)
        ax.plot(x_grid, y_opt, color="orangered", linewidth=3, label="Optimal")

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.show()


def train_spline_GDRO(
    N_envs,
    Y_envs,
    Omega,
    lr_beta=1e-3,
    eta=0.1,
    lambda_reg=0.01,
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
      N_envs: list of [n_e x M] torch.Tensor design (B-spline basis) matrices.
      Y_envs: list of [n_e x 1] torch.Tensor target vectors.
      Omega:  [M x M] torch.Tensor penalty matrix.
      lr_beta: stepsize for beta-descent.
      eta: mirror-ascent stepsize for environment-weights.
      lambda_reg: regularization coefficient.
      epochs: max number of epochs.
      early_stopping: whether to use early stopping.
      patience: epochs with no improvement before stopping.
      min_delta: minimum improvement to reset patience.
      verbose: print progress.
      seed: random seed for reproducibility.

    Returns:
      beta: numpy array of shape (M,) final coefficients.
      p: numpy array of length E final environment weights.
    """
    torch.manual_seed(seed)
    E = len(N_envs)
    M = Omega.shape[0]

    # Initialize beta (unconstrained)
    beta = torch.zeros(M, requires_grad=True)
    optimizer = optim.Adam([beta], lr=lr_beta, weight_decay=weight_decay)

    # Initialize environment weights p on the simplex
    p = torch.ones(E, dtype=torch.float32) / E

    best_obj = float("inf")
    best_beta = None
    best_p = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Compute per-environment MSE losses
        losses = []
        for e in range(E):
            N_e = N_envs[e]  # [n_e x M]
            Y_e = Y_envs[e]  # [n_e x 1]
            pred_e = N_e @ beta  # [n_e]
            loss_e = F.mse_loss(pred_e, Y_e.squeeze(), reduction="mean")
            losses.append(loss_e)
        losses = torch.stack(losses)  # [E]

        # Mirror ascent: update p
        with torch.no_grad():
            p = p * torch.exp(eta * losses)
            p = p / p.sum()

        # Compose objective: weighted loss + reg
        weighted_loss = torch.dot(p, losses)
        reg = lambda_reg * (beta @ (Omega @ beta))
        obj = weighted_loss + reg

        # Gradient descent step on beta
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        # Early stopping check
        current = obj.item()
        if current + min_delta < best_obj:
            best_obj = current
            best_beta = beta.detach().cpu().numpy().copy()
            best_p = p.detach().cpu().numpy().copy()
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % (epochs // 10 or 1) == 0:
            print(f"Epoch {epoch}: obj={current:.6f}, best={best_obj:.6f}")

        if early_stopping and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Fallback if no improvement ever
    if best_beta is None:
        best_beta = beta.detach().cpu().numpy()
        best_p = p.detach().cpu().numpy()

    return best_beta, best_p


def project_onto_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection of a vector v onto the probability simplex {p: p>=0, sum(p)=1}.
    Implementation based on sorting method.
    """
    # Sort v in descending order
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    rho = torch.where(
        u * torch.arange(1, len(u) + 1, device=v.device) > (cssv - 1)
    )[0].max()
    theta = (cssv[rho] - 1) / (rho + 1).float()
    w = torch.clamp(v - theta, min=0)
    return w


def train_spline_extragradient(
    N_envs,
    Y_envs,
    Omega,
    alpha=1e-3,
    lambda_reg=0.01,
    epochs=500,
    seed=0,
    verbose=False,
):
    """
    Extragradient method for min_beta max_p L(beta,p) where
      L(beta,p) = sum_e p_e * MSE_e(beta) + lambda_reg * beta^T Omega beta,
    and p lies on the probability simplex.

    Returns:
      beta: torch.Tensor of shape [M]
      p:    torch.Tensor of shape [E]
    """
    torch.manual_seed(seed)
    E = len(N_envs)
    M = Omega.shape[0]

    # Initialize variables
    beta = torch.zeros(M, dtype=Omega.dtype, requires_grad=False)
    p = torch.ones(E, dtype=Omega.dtype) / E

    for t in range(epochs):
        # Compute gradients at (beta, p)
        # f_e = MSE loss for env e
        losses = []
        grads = torch.zeros_like(beta)
        for e in range(E):
            N_e = N_envs[e]
            Y_e = Y_envs[e].squeeze()
            n_e = N_e.shape[0]
            pred = N_e @ beta
            residual = pred - Y_e
            loss_e = residual.pow(2).mean()
            losses.append(loss_e)
            # gradient wrt beta for f_e
            grads += p[e] * (2.0 / n_e) * (N_e.t() @ residual)
        losses = torch.stack(losses)  # [E]
        # add reg gradient
        grads += 2 * lambda_reg * (Omega @ beta)

        # Extragradient half-step
        beta_half = beta - alpha * grads
        p_hat = project_onto_simplex(p + alpha * losses)

        # Compute gradients at (beta_half, p_hat)
        losses_half = []
        grads_half = torch.zeros_like(beta)
        for e in range(E):
            N_e = N_envs[e]
            Y_e = Y_envs[e].squeeze()
            n_e = N_e.shape[0]
            pred_h = N_e @ beta_half
            resid_h = pred_h - Y_e
            loss_e_h = resid_h.pow(2).mean()
            losses_half.append(loss_e_h)
            grads_half += p_hat[e] * (2.0 / n_e) * (N_e.t() @ resid_h)
        losses_half = torch.stack(losses_half)
        grads_half += 2 * lambda_reg * (Omega @ beta_half)

        # Extragradient full-step
        beta = beta - alpha * grads_half
        p = project_onto_simplex(p + alpha * losses_half)

        if verbose and (t % max(1, epochs // 10) == 0):
            obj = (p * losses).sum() + lambda_reg * beta @ (Omega @ beta)
            print(f"Iter {t}/{epochs}: obj={obj.item():.6f}")

    return beta.detach().cpu().numpy(), p.detach().cpu().numpy()


def train_spline_online_GDRO(
    N_envs: List[torch.Tensor],
    Y_envs: List[torch.Tensor],
    Omega: torch.Tensor,
    eta: float = 0.1,
    eta_theta: float = 1e-3,
    lambda_reg: float = 0.01,
    T: int = 10000,
    batch_size: int = 1,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Online GDRO for spline coefficients (theta) and group weights (q).
    At each step, sample a group g uniformly,
    sample a batch from that group, update q_g via multiplicative weights,
    then update theta via weighted gradient descent.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    E = len(N_envs)
    M = Omega.shape[0]

    # Initialize theta and q
    theta = torch.zeros(M, device=device, dtype=Omega.dtype)
    q = torch.ones(E, device=device, dtype=Omega.dtype) / E

    # Create DataLoaders for each environment
    loaders = []
    for N_e, Y_e in zip(N_envs, Y_envs):
        # each dataset: inputs are basis rows, targets
        ds = TensorDataset(N_e.to(device), Y_e.to(device))
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
    iters = [iter(loader) for loader in loaders]

    for t in range(1, T + 1):
        # 1) Sample group g uniformly
        g = random.randrange(E)
        # 2) Sample batch (N_batch, y_batch) from group g
        try:
            N_batch, y_batch = next(iters[g])
        except StopIteration:
            iters[g] = iter(loaders[g])
            N_batch, y_batch = next(iters[g])
        y_batch = y_batch.view(-1)

        # 3) Compute loss and gradient for this batch
        pred = N_batch @ theta
        resid = pred - y_batch
        loss = resid.pow(2).mean()
        # gradient of loss w.r.t theta: (2/n) * N_batch^T resid
        grad_loss = (2.0 / N_batch.shape[0]) * (N_batch.t() @ resid)

        # 4) Mirror ascent on q_g
        # update only the g-th coordinate multiplicatively
        q_g_prime = q[g] * torch.exp(eta * loss.detach())
        q[g] = q_g_prime
        # renormalize entire q
        q = q / q.sum()

        # 5) Gradient descent on theta with weighted loss + reg
        # include regularizer gradient
        grad_reg = 2 * lambda_reg * (Omega @ theta)
        theta = theta - eta_theta * (q[g] * grad_loss + grad_reg)

        # optional logging
        if verbose and t % (T // 10 or 1) == 0:
            worst_idx = torch.argmax(q)
            print(
                f"Iter {t}/{T}: sampled g={g}, q_max_idx={worst_idx.item()}, loss={loss.item():.4f}, theta_norm={theta.norm().item():.4f}"
            )

    return theta.detach().cpu().numpy(), q.detach().cpu().numpy()


def train_spline_subgradient(
    N_envs: List[torch.Tensor],
    Y_envs: List[torch.Tensor],
    Omega: torch.Tensor,
    lr: float = 1e-2,
    lambda_reg: float = 1e-3,
    epochs: int = 1000,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subgradient descent on β for min_β [ max_e MSE_e(β) ] + λ βᵀΩβ.
    Returns β and history of worst‐case losses.
    """
    N_envs = [N_e.to(device) for N_e in N_envs]
    Y_envs = [Y_e.to(device) for Y_e in Y_envs]
    Omega = Omega.to(device)

    E = len(N_envs)
    M = Omega.shape[0]

    # init β
    beta = torch.zeros(M, device=device)

    history = []

    for k in range(1, epochs + 1):
        # 1) compute per‐env MSE and record losses
        losses = torch.zeros(E, device=device)
        for e in range(E):
            N_e = N_envs[e]
            y_e = Y_envs[e]
            pred = N_e @ beta
            losses[e] = F.mse_loss(pred, y_e, reduction="mean")

        # 2) pick worst‐case environment
        e_star = int(losses.argmax().item())
        f_star = losses[e_star].item()
        history.append(f_star)

        # 3) compute subgradient of f_{e*}
        N_e = N_envs[e_star]
        y_e = Y_envs[e_star]
        n_e = N_e.shape[0]

        # ∇_β MSE = (2/n_e) N_eᵀ (N_e β − y_e)
        grad_data = (2.0 / n_e) * (N_e.t() @ (N_e @ beta - y_e))

        # ∇_β reg = 2 λ Ω β
        grad_reg = 2 * lambda_reg * (Omega @ beta)

        gk = grad_data + grad_reg

        # 4) subgradient descent update
        beta = beta - lr * gk

        if verbose and (k % (epochs // 10 or 1) == 0):
            obj = f_star + lambda_reg * (beta @ (Omega @ beta)).item()
            print(
                f"iter {k:4d}  env*={e_star}  MSE*={f_star:.4e}  obj≈{obj:.4e}"
            )

    return beta.detach().cpu().numpy(), torch.tensor(history)
