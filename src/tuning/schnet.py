from .base import BaseTuner
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.schnet import SchNetRegressor
from src.utils.metrics import compute_metrics
from .registry import TuningRegistry



@TuningRegistry.register("schnet")
class SchNetTuner(BaseTuner):

    def __init__(self, train_ds, val_ds, epochs=10, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs=epochs, epochs_trials=epochs_trials, device=device)

    # create model
    def create_model_from_params(self, params):
        return SchNetRegressor(
            hidden_channels=params["hidden_channels"],
            num_filters=params["num_filters"],
            num_interactions=params["num_interactions"]
        ).to(self.device)  # later cutoff

    # ---------------------------------------------------------------------
    # Training / evaluation function (SchNet-specific)
    # ---------------------------------------------------------------------
    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0

        # --- IMPORTANT ---
        # Use torch.no_grad() only when NOT training
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                batch = batch.to(self.device)

                if train:
                    optimizer.zero_grad()

                out = model(batch.z, batch.pos, batch.batch)
                pred = out.squeeze(-1)
                target = batch.y.squeeze(-1)

                loss = criterion(pred, target)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    def create_model(self, trial, hidden_channels_opts, num_filters_opts,
                     num_interactions_low, num_interactions_high):

        hidden_channels = trial.suggest_categorical("hidden_channels", hidden_channels_opts)
        num_filters = trial.suggest_categorical("num_filters", num_filters_opts)
        num_interactions = trial.suggest_int("num_interactions",
                                             num_interactions_low,
                                             num_interactions_high)

        return SchNetRegressor(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions
        ).to(self.device)



    # objective
    def objective(self, trial, **kwargs):

        # SchNet-specific params
        hidden_channels_opts = kwargs.get("hidden_channels_opts", [32, 64])
        num_filters_opts = kwargs.get("num_filters_opts", [32, 64])
        num_interactions_low = kwargs.get("num_interactions_low", 1)
        num_interactions_high = kwargs.get("num_interactions_high", 5)

        # General parameters
        batch_size_opts = kwargs.get("batch_size_opts", [16])
        lr_low = kwargs.get("lr_low", 1e-4)
        lr_high = kwargs.get("lr_high", 1e-2)

        # Optuna sampling
        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)

        train_loader, val_loader = self.create_loaders(batch_size)

        model = self.create_model(
            trial,
            hidden_channels_opts,
            num_filters_opts,
            num_interactions_low,
            num_interactions_high
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=2,
            min_lr=1e-6
        )

        # ---- tuning loop ----
        for epoch in range(self.epochs_trials):
            self.run_epoch(True,  train_loader, model, criterion, optimizer)
            val_loss = self.run_epoch(False, val_loader, model, criterion)

            scheduler.step(val_loss)

        # ---- compute metrics at the end ----
        y_true, y_pred = self.get_predictions(val_loader, model)
        metrics = compute_metrics(y_true, y_pred)
        trial.set_user_attr("metrics", metrics)

        return val_loss

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    def get_predictions(self, loader, model):
        model.eval()

        preds, trues = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                y_hat = model(batch.z, batch.pos, batch.batch).squeeze(-1)
                y = batch.y.squeeze(-1).float()

                preds.append(y_hat.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds)
        trues = torch.cat(trues)
        return trues, preds







