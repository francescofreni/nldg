# code modified from https://github.com/zywang0701/DRoL/blob/main/methods/others.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from nldg.utils import set_all_seeds
from torch.utils.data import TensorDataset, DataLoader


class GroupDRO:
    def __init__(
        self, data, hidden_dims, seed=42, risk="mse", oracle_kwargs=None
    ):
        set_all_seeds(seed)
        self.data = data
        self.hidden_dims = list(hidden_dims)
        self.risk = risk
        self.oracle_kwargs = oracle_kwargs or {}
        self._regret_baselines = None  # torch.tensor of shape [L]
        self.group_predictors = []
        # Model for GroupDRO
        self.model = self._make_model()

    def _make_model(self):
        layers, prev_dim = [], self.data.d
        for h in self.hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _group_y2_means(self, Y_groups):
        # for negative reward: E[Y^2] per group (scalar per group)
        return torch.tensor(
            [torch.mean(y.squeeze() ** 2).item() for y in Y_groups],
            dtype=torch.float32,
        )

    def _fit_group_predictors(self, X_groups, Y_groups):
        """
        Train predictors on the different environments
        """
        baselines = []
        group_predictors = []
        for l in range(self.data.L):
            X, Y = X_groups[l], Y_groups[l]
            n = X.shape[0]
            idx = torch.randperm(n)
            n_val = max(1, int(0.2 * n))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            Xtr, Ytr = X[tr_idx], Y[tr_idx]
            Xval, Yval = X[val_idx], Y[val_idx]

            model = self._make_model()
            opt = optim.Adam(
                model.parameters(),
                lr=self.oracle_kwargs.get("lr", 1e-3),
                weight_decay=self.oracle_kwargs.get("weight_decay", 1e-5),
            )
            loss_fn = nn.MSELoss()
            best = float("inf")
            best_state = None
            patience = self.oracle_kwargs.get("early_stopping_patience", 5)
            min_delta = self.oracle_kwargs.get("min_delta", 1e-4)
            no_improve = 0
            epochs = self.oracle_kwargs.get("epochs", 100)

            for _ in range(epochs):
                model.train()
                opt.zero_grad()
                preds = model(Xtr)
                loss = loss_fn(preds, Ytr)
                loss.backward()
                opt.step()

                # val
                model.eval()
                val_pred = model(Xval).detach()
                val_loss = loss_fn(val_pred, Yval).item()
                if val_loss < best - min_delta:
                    best, best_state, no_improve = (
                        val_loss,
                        {k: v.clone() for k, v in model.state_dict().items()},
                        0,
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            baselines.append(best)
            group_predictors.append(model)

        self.group_predictors = group_predictors
        return torch.tensor(baselines, dtype=torch.float32)

    def fit(
        self,
        lr_model=1e-3,
        eta=0.1,
        epochs=100,
        weight_decay=1e-5,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=5,
        min_delta=1e-4,
        verbose=False,
        device=None,
    ):
        """
        Trains the model using Group DRO with Mirror Ascent for group weights.
        Adds optional early stopping based on improvements in weighted loss.

        Args:
            lr_model (float, optional): Learning rate for the model optimizer. Defaults to 1e-3.
            eta (float, optional): Learning rate for mirror ascent. Defaults to 0.1.
            epochs (int, optional): Max number of training epochs. Defaults to 100.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
            batch_size (int, optional): Batch size. Defaults to 128.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): # epochs to wait for improvement. Defaults to 5.
            min_delta (float, optional): Minimum improvement to reset patience. Defaults to 1e-4.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
            device (torch.device, optional): Device to use. Defaults to None.
        """
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.model.to(device)

        # Optimizer for model parameters
        optimizer_model = optim.Adam(
            self.model.parameters(), lr=lr_model, weight_decay=weight_decay
        )
        # Loss function (Mean Squared Error)
        loss_fn = nn.MSELoss(reduction="none")
        # Initialize group weights
        L = self.data.L
        weights = torch.ones(L, device=device) / L

        self.model.train()

        # Convert all data to PyTorch tensors
        X_tensors = []
        Y_tensors = []
        G_tensors = []
        for g_idx, (Xg, Yg) in enumerate(
            zip(self.data.X_sources_list, self.data.Y_sources_list)
        ):
            X_tensors.append(torch.tensor(Xg, dtype=torch.float32))
            Y_tensors.append(
                torch.tensor(Yg, dtype=torch.float32).unsqueeze(1)
            )
            G_tensors.append(torch.full((len(Xg),), g_idx, dtype=torch.long))

        X_all = torch.cat(X_tensors, dim=0)
        Y_all = torch.cat(Y_tensors, dim=0)
        G_all = torch.cat(G_tensors, dim=0)

        dataset = TensorDataset(X_all, Y_all, G_all)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Precompute baselines for reward / regret
        if self.risk == "reward":
            # E[Y^2] per group, same as before
            Y_groups = [
                torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
                for Y in self.data.Y_sources_list
            ]
            y2_means = self._group_y2_means(Y_groups).to(device)  # [L]
        elif self.risk == "regret":
            X_groups = [
                torch.tensor(X, dtype=torch.float32)
                for X in self.data.X_sources_list
            ]
            Y_groups = [
                torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
                for Y in self.data.Y_sources_list
            ]
            if self._regret_baselines is None:
                self._regret_baselines = self._fit_group_predictors(
                    X_groups, Y_groups
                )
            b = self._regret_baselines.to(device)  # [L]

        # Track best state for early stopping
        best_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_weights = weights.detach().clone()
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for Xb, Yb, Gb in loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                Gb = Gb.to(device)  # group indices in this batch

                preds = self.model(Xb)  # [B, 1]
                per_sample_loss = loss_fn(preds, Yb).view(-1)  # [B]

                # Subtract group-specific baseline if needed
                if self.risk == "reward":
                    baseline_per_sample = y2_means[Gb]  # [B]
                    per_sample_loss = per_sample_loss - baseline_per_sample
                elif self.risk == "regret":
                    baseline_per_sample = b[Gb]  # [B]
                    per_sample_loss = per_sample_loss - baseline_per_sample

                # Compute per-group average loss within this batch
                group_losses = torch.zeros(L, device=device)
                group_counts = torch.zeros(L, device=device)

                for g_idx in range(L):
                    mask = Gb == g_idx
                    if mask.any():
                        group_losses[g_idx] = per_sample_loss[mask].mean()
                        group_counts[g_idx] = mask.sum()

                # Mirror Ascent update: only groups with count>0 get updated,
                # others effectively see loss 0 (exp(0)=1 => unchanged weight).
                adjusted_losses = (
                    group_losses  # you could add adj here if desired
                )
                new_weights = weights * torch.exp(eta * adjusted_losses)
                new_weights = new_weights / new_weights.sum()
                weights = new_weights.detach()

                # Weighted batch loss (only groups that appear contribute)
                weighted_loss = (weights * group_losses).sum()

                optimizer_model.zero_grad()
                weighted_loss.backward()
                optimizer_model.step()

                epoch_losses.append(weighted_loss.item())

            # Epoch-level stats for early stopping
            current_loss = (
                float(np.mean(epoch_losses)) if epoch_losses else 0.0
            )

            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_weights = weights.detach().clone()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs}, "
                    f"Mean Weighted Loss: {current_loss:.6f}, Best: {best_loss:.6f}"
                )

            if early_stopping and no_improve_epochs >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement in {early_stopping_patience} epochs)."
                    )
                break

        # Restore best model and weights
        self.model.load_state_dict(best_model_state)
        self.final_group_weights_ = best_weights.clone().cpu()

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
            X_tensor = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_tensor).numpy().flatten()
        return preds

    def predict_per_group(self, X, E):
        preds = np.zeros(len(E))
        for l in range(self.data.L):
            model = self.group_predictors[l]
            model.eval()
            mask = E == l
            with torch.no_grad():
                X_tensor = torch.tensor(X[mask], dtype=torch.float32)
                preds[mask] = model(X_tensor).numpy().flatten()
        return preds
