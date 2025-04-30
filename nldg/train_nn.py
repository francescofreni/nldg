import torch
import torch.optim as optim
import torch.nn as nn
from nldg.nn import NN
from nldg.utils import set_all_seeds
import numpy as np
import copy

SEED = 42


def train_model(
    X_train,
    Y_train,
    hidden_dims=[64, 64],
    lr=1e-3,
    weight_decay=1e-4,
    epochs=100,
    early_stopping=True,
    patience=5,
    min_delta=1e-4,
    verbose=False,
    seed=SEED,
):
    set_all_seeds(seed)
    model = NN(input_dim=X_train.shape[1], hidden_dims=hidden_dims)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = model.state_dict().copy()
    no_improve = 0

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, Y_tensor)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"[Default] Epoch {epoch}, Loss: {train_loss:.6f}")

        # Early stopping
        if early_stopping:
            if train_loss + min_delta < best_loss:
                best_loss = train_loss
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"[Default] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def predict_default(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
    return preds


def train_model_GDRO(
    X_train,
    Y_train,
    E_train,
    hidden_dims=[64, 64],
    lr_model=1e-3,
    eta=0.1,
    epochs=100,
    weight_decay=1e-5,
    early_stopping=True,
    early_stopping_patience=5,
    min_delta=1e-4,
    verbose=False,
    seed=SEED,
):
    # Initialize model
    set_all_seeds(seed)
    model = NN(input_dim=X_train.shape[1], hidden_dims=hidden_dims)
    optimizer = optim.Adam(
        model.parameters(), lr=lr_model, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss(reduction="mean")

    n_envs = len(np.unique(E_train))
    weights = torch.ones(n_envs) / n_envs

    model.train()

    X_groups = [
        torch.tensor(X_train[E_train == e], dtype=torch.float32)
        for e in np.unique(E_train)
    ]
    Y_groups = [
        torch.tensor(Y_train[E_train == e], dtype=torch.float32).unsqueeze(1)
        for e in np.unique(E_train)
    ]

    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_weights = weights.detach().clone()
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        group_losses = []

        # Compute loss for each group
        for i, e in enumerate(np.unique(E_train)):
            X = X_groups[i]
            Y = Y_groups[i]
            preds = model(X)

            loss = loss_fn(preds, Y)  # - torch.mean(Y**2)
            group_losses.append(loss)

        group_losses_tensor = torch.stack(group_losses)  # shape: [num_groups]

        # Mirror Ascent: update group weights
        new_weights = weights * torch.exp(eta * group_losses_tensor)
        new_weights = new_weights / new_weights.sum()
        weights = new_weights.detach()

        # Compute the weighted loss
        weighted_loss = torch.dot(weights, group_losses_tensor)

        # Backpropagation step
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        current_loss = weighted_loss.item()

        # Early stopping logic
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_weights = weights.detach().clone()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if verbose and (epoch == 1 or epoch % max(1, epochs // 20) == 0):
            print(
                f"Epoch {epoch}/{epochs}, Weighted Loss: {current_loss:.6f}, Best: {best_loss:.6f}"
            )

        if early_stopping and no_improve_epochs >= early_stopping_patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch} (no improvement in {early_stopping_patience} epochs)."
                )
            break

    # Restore best model weights and group weights
    model.load_state_dict(best_model_state)
    return model, best_weights


def train_model_GDRO_online(
    X_train,
    Y_train,
    E_train,
    hidden_dims=[64, 64],
    lr_model=1e-3,
    eta=0.1,
    epochs=100,
    batch_size=1,
    weight_decay=1e-5,
    seed=SEED,
    verbose=False,
):
    """
    Online mirror-ascent GDRO (Algorithm 1): sample one group and one example per step.

    Args:
        X_train (np.ndarray): training features.
        Y_train (np.ndarray): training targets.
        E_train (np.ndarray): group assignments.
        hidden_dims (list): hidden layer sizes.
        lr_model (float): learning rate for model update (eta_theta).
        eta (float): step size for q-update (eta_q).
        steps (int): total online iterations T.
        batch_size (int): examples per step from selected group.
        weight_decay (float): L2 penalty.
        seed (int): random seed.
        verbose (bool): print periodic loss.

    Returns:
        model (nn.Module): trained NN.
        q (torch.Tensor): final group weights.
    """
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr_model, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss(reduction="mean")

    # Initialize q
    groups = np.unique(E_train)
    m = len(groups)
    q = torch.ones(m, device=device) / m

    # Pre-split indices per group
    group_indices = {
        i: np.where(E_train == g)[0] for i, g in enumerate(groups)
    }

    model.train()
    for t in range(1, epochs + 1):
        # 1) sample group index i
        i = np.random.randint(0, m)
        idxs = np.random.choice(
            group_indices[i], size=batch_size, replace=True
        )

        Xb = torch.tensor(X_train[idxs], dtype=torch.float32, device=device)
        Yb = torch.tensor(
            Y_train[idxs], dtype=torch.float32, device=device
        ).unsqueeze(1)

        # 2) compute loss on batch
        preds = model(Xb)
        loss = loss_fn(preds, Yb)

        # 3) update q_i via mirror ascent
        with torch.no_grad():
            q_i = q[i] * torch.exp(eta * loss.detach())
            q[i] = q_i
            q /= q.sum()

        # 4) weighted gradient step on theta
        optimizer.zero_grad()
        (q[i] * loss).backward()
        optimizer.step()

        if verbose and t % (epochs // 10 or 1) == 0:
            print(
                f"[Online GDRO] Step {t}/{epochs}, Loss={loss.item():.4f}, q_max={q.max().item():.3f}"
            )

    return model, q


# Example predict function (assuming single array input)
def predict_GDRO(model, X):
    model.eval()  # set to evaluation mode
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
    return preds
