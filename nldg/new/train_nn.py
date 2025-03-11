import torch
import torch.optim as optim
import torch.nn as nn
from nldg.new.nn import NN
from nldg.new.utils import set_all_seeds
import numpy as np

SEED = 42


def max_mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    envs: torch.Tensor
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
        mask = (envs == env)
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
        default: bool = False
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
