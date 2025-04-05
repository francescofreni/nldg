import torch
import torch.optim as optim
import torch.nn as nn
from nldg.new.archive.nn import NN, NN_GDRO
from nldg.new.archive.utils import set_all_seeds
import numpy as np
import copy

SEED = 42


def max_mse_loss(
    preds: torch.Tensor, targets: torch.Tensor, envs: torch.Tensor
) -> torch.Tensor:
    """
    Computes the maximum Mean Squared Error (MSE) across different environments.

    Args:
        preds: Tensor of model predictions.
        targets: Tensor of true target values.
        envs: Tensor of environment labels.

    Returns:
        The maximum MSE loss among all environments as a scalar tensor.
    """
    unique_envs = torch.unique(envs)
    losses = []
    mse = nn.MSELoss()

    for env in unique_envs:
        mask = envs == env
        if mask.sum() > 0:
            mse_env = mse(preds[mask], targets[mask])
            losses.append(mse_env)

    return max(losses) if losses else torch.tensor(0.0)


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    E_train: np.ndarray,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    default: bool = False,
) -> torch.nn.Module:
    """
    Trains a neural network on the given training data.

    Args:
        X_train: Features in the training dataset.
        Y_train: Response in the training dataset.
        E_train: Environment labels in the training datasets.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        verbose: Whether to print training progress every 10 epochs .
        default: If True, uses standard MSE loss; otherwise, uses max-MSE loss across environments.

    Returns:
        The trained neural network model.
    """
    set_all_seeds(SEED)
    model = NN(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if default:
        criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    E_tensor = torch.tensor(E_train, dtype=torch.int64)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        if default:
            loss = criterion(preds, Y_tensor)
        else:
            loss = max_mse_loss(preds, Y_tensor, E_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    return model


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
):
    # Initialize model
    model = NN_GDRO(input_dim=X_train.shape[1], hidden_dims=hidden_dims)
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

            loss = loss_fn(preds, Y) - torch.mean(Y**2)
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


# Example predict function (assuming single array input)
def predict_GDRO(model, X):
    """
    Makes predictions using the trained model.

    Args:
        model: Trained neural network.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Predicted values.
    """
    model.eval()  # set to evaluation mode
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
    return preds
