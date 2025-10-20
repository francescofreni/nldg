import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from nldg.utils import set_all_seeds
import matplotlib.pyplot as plt
from nldg.rf import MaggingRF
from adaXT.random_forest import RandomForest
from tqdm import tqdm

N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 30
RANDOM_STATE = 42
N_PER_ENV = 400
NOISE_STD = 0.5
N_SIM = 50


class DataContainer:
    def __init__(self, n=300, noise_std=0.2):
        self.n = n  # number of samples in each source domain
        self.d = 1  # number of features
        self.L = None  # number of source domains
        self.noise_std = noise_std

        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.E_sources_list = []  # list of source environment labels

        self.f_funcs = []  # list of source conditional outcome functions

    def generate_funcs_list(self, L):
        self.L = L
        f1 = lambda x: np.where(x <= 0, -0.5 * x, 4 * x)
        f2 = lambda x: np.where(x <= 0, 3 * x, 0.5 * x)
        f3 = lambda x: np.where(x <= 0, 2.5 * x, x)
        self.f_funcs = [f1, f2, f3]

    def generate_data(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._reset_lists()

        X = self.rng.uniform(-4, 4, size=self.n)
        for l in range(self.L):
            Y = self.f_funcs[l](X) + self.rng.normal(
                0, self.noise_std, size=self.n
            )
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
            self.E_sources_list.append(np.full(self.n, l, dtype=int))

    def _reset_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        self.E_sources_list = []


class GroupDRO:
    def __init__(self, data, hidden_dims, seed=42):
        """
        Args:
            data (_type_): DataContainer
            input_dim (int): Number of features.
            hidden_dims (List[int], optional): Sizes of hidden layers.
            seed (int, optional): Random seed.
        """
        set_all_seeds(seed)
        self.data = data
        # Define the neural network architecture
        layers = []
        prev_dim = self.data.d
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)

    def fit(
        self,
        lr_model=1e-3,
        eta=0.1,
        epochs=100,
        weight_decay=1e-5,
        early_stopping=True,
        early_stopping_patience=5,
        min_delta=1e-4,
        verbose=False,
    ):
        """
        Trains the model using Group DRO with Mirror Ascent for group weights.
        Adds optional early stopping based on improvements in weighted loss.

        Args:
            lr_model (float, optional): Learning rate for the model optimizer. Defaults to 1e-3.
            eta (float, optional): Learning rate for mirror ascent. Defaults to 0.1.
            epochs (int, optional): Max number of training epochs. Defaults to 100.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): # epochs to wait for improvement. Defaults to 5.
            min_delta (float, optional): Minimum improvement to reset patience. Defaults to 1e-4.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
        """
        # Optimizer for model parameters
        optimizer_model = optim.Adam(
            self.model.parameters(), lr=lr_model, weight_decay=weight_decay
        )
        # Loss function (Mean Squared Error)
        loss_fn = nn.MSELoss(reduction="mean")  # Compute mean loss per group
        # Initialize group weights
        weights = torch.ones(self.data.L) / self.data.L

        self.model.train()
        # Convert all data to PyTorch tensors
        X_groups = [
            torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            for X in self.data.X_sources_list
        ]
        Y_groups = [
            torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
            for Y in self.data.Y_sources_list
        ]

        # Track best state for early stopping
        best_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_weights = weights.detach().clone()
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            group_losses = []

            # compute loss for each group
            for l in range(self.data.L):
                X = X_groups[l]
                Y = Y_groups[l]
                preds = self.model(X)
                loss = loss_fn(preds, Y)
                group_losses.append(loss)

            group_losses_tensor = torch.stack(
                group_losses
            )  # shape: [num_groups]

            # Mirror Ascent: update group weights
            new_weights = weights * torch.exp(eta * group_losses_tensor)
            new_weights = new_weights / new_weights.sum()
            weights = new_weights.detach()

            # Weighted loss
            weighted_loss = torch.dot(weights, group_losses_tensor)

            # Backprop and update model parameters
            optimizer_model.zero_grad()
            weighted_loss.backward()
            optimizer_model.step()

            # Convert to float
            current_loss = weighted_loss.item()
            # Early stopping logic
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
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
        self.model.load_state_dict(best_model_state)
        weights = best_weights.clone()

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted continuous outcomes.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            preds = self.model(X_tensor).numpy().flatten()
        return preds


def plot_results(data, X_sorted, Y_rf, Y_maxrmrf, Y_magging, Y_gdro):
    data_colors = ["black", "grey", "silver"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in range(data.L):
        ax.scatter(
            data.X_sources_list[L],
            data.Y_sources_list[L],
            color=data_colors[L],
            marker="o",
            alpha=0.5,
            s=30,
            label=f"Env {L + 1}",
        )

    ax.plot(
        X_sorted,
        Y_rf,
        color="#5790FC",
        linewidth=2,
        label="RF",
    )
    ax.plot(
        X_sorted,
        Y_maxrmrf,
        color="#F89C20",
        linewidth=2,
        label="MaxRM-RF(mse)",
    )
    ax.plot(
        X_sorted,
        Y_magging,
        color="#964A8B",
        linewidth=2,
        label="RF(magging)",
    )
    ax.plot(
        X_sorted,
        Y_gdro,
        color="#86C8DD",
        linewidth=2,
        label="GroupDRO-NN(mse)",
    )

    x_range = np.linspace(X_sorted.min(), X_sorted.max(), 1000)
    y_opt = np.where(x_range > 0, 2.25 * x_range, 1.25 * x_range)
    ax.plot(
        x_range,
        y_opt,
        color="#E42536",
        linewidth=3,
        label="Oracle",
        linestyle="--",
    )

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.grid(True, linewidth=0.2, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(
        script_dir, "..", "..", "results", "output_additional"
    )
    os.makedirs(parent_dir, exist_ok=True)
    outpath = os.path.join(parent_dir, "intro.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    # plt.show()


def f_opt(x):
    return np.where(x <= 0, 1.25 * x, 2.25 * x)


def one_sim_step(seed, ret_ise=True):
    data.generate_data(seed=seed)

    Xtr = np.concatenate(data.X_sources_list)
    Ytr = np.concatenate(data.Y_sources_list)
    Etr = np.concatenate(data.E_sources_list)
    Xtr_sorted = np.sort(data.X_sources_list[0], axis=0)

    # RF
    rf = RandomForest(
        "Regression",
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        seed=RANDOM_STATE,
    )
    rf.fit(Xtr, Ytr)
    preds_rf = rf.predict(Xtr_sorted)
    preds_rf_grid = rf.predict(x_grid)
    ise_rf = np.sum((preds_rf_grid - preds_opt) ** 2) * delta

    # MaxRM-RF
    rf.modify_predictions_trees(Etr)
    preds_maxrmrf = rf.predict(Xtr_sorted)
    preds_maxrmrf_grid = rf.predict(x_grid)
    ise_maxrmrf = np.sum((preds_maxrmrf_grid - preds_opt) ** 2) * delta

    # Magging-RF
    rf_magging = MaggingRF(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        backend="adaXT",
    )
    _ = rf_magging.fit(Xtr, Ytr, Etr)
    preds_magging = rf_magging.predict(Xtr_sorted)
    preds_magging_grid = rf_magging.predict(x_grid)
    ise_magging = np.sum((preds_magging_grid - preds_opt) ** 2) * delta

    # Group DRO
    gdro = GroupDRO(data, hidden_dims=[64], seed=RANDOM_STATE)
    gdro.fit(epochs=1000, lr_model=0.01, eta=0.01, weight_decay=0.01)
    preds_gdro = gdro.predict(Xtr_sorted)
    preds_gdro_grid = gdro.predict(x_grid)
    ise_gdro = np.sum((preds_gdro_grid - preds_opt) ** 2) * delta

    if ret_ise:
        return ise_rf, ise_maxrmrf, ise_magging, ise_gdro
    else:
        return Xtr_sorted, preds_rf, preds_maxrmrf, preds_magging, preds_gdro


if __name__ == "__main__":
    x_min, x_max = -4, 4
    n_grid = 100
    x_grid = np.linspace(-4, 4, n_grid)
    delta = (x_max - x_min) / n_grid
    preds_opt = f_opt(x_grid).ravel()

    data = DataContainer(n=N_PER_ENV, noise_std=NOISE_STD)
    data.generate_funcs_list(L=3)
    ise = np.zeros((N_SIM, 4))

    for i in tqdm(range(N_SIM)):
        ise[i, 0], ise[i, 1], ise[i, 2], ise[i, 3] = one_sim_step(
            seed=RANDOM_STATE + i
        )

    names = ["RF", "MaxRM-RF(mse)", "RF(magging)", "GroupDRO-NN(mse)"]
    means = ise.mean(axis=0)
    stds = ise.std(axis=0, ddof=1)
    z = 1.96
    halfwidth = z * stds / np.sqrt(N_SIM)
    print("\nMean ISE (±95% CI):")
    for name, m, h in zip(names, means, halfwidth):
        print(f"{name:>16}: {m:.6f}  [{(m - h):.6f}, {(m + h):.6f}]")

    # Repeat to make the plot
    (
        Xtr_sorted,
        preds_rf,
        preds_maxrmrf,
        preds_magging,
        preds_gdro,
    ) = one_sim_step(seed=RANDOM_STATE, ret_ise=False)

    plot_results(
        data, Xtr_sorted, preds_rf, preds_maxrmrf, preds_magging, preds_gdro
    )
