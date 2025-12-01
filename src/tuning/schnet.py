from .base import BaseTuner
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.schnet import SchNetRegressor
from .registry import TuningRegistry
from src.training.schnet import SchNetTrainer



@TuningRegistry.register("schnet")
class SchNetTuner(BaseTuner):

    trainer_cls = SchNetTrainer

    def __init__(self, train_ds, val_ds, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs_trials=epochs_trials, device=device)

    

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
        num_interactions_low = kwargs.get("lr", {}).get("low", 1)
        num_interactions_high = kwargs.get("lr", {}).get("high", 5)

        # General parameters
        batch_size_opts = kwargs.get("batch_size_opts", [16])
        lr_low = kwargs.get("lr", {}).get("low", 1e-4)
        lr_high = kwargs.get("lr", {}).get("high", 1e-2)

        # Optuna sampling
        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)

        # trainer
        trainer = self.trainer_cls(self.train_ds, self.val_ds) # maybe pass device if needed

        # loaders
        train_loader, val_loader = trainer.create_loaders(batch_size)

        # model
        model = self.create_model(
            trial,
            hidden_channels_opts,
            num_filters_opts,
            num_interactions_low,
            num_interactions_high
        )

        # optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=2,
            min_lr=1e-6
        )

        # ---- tuning loop ----
        for epoch in range(self.epochs_trials):
            trainer.run_epoch(True,  train_loader, model, criterion, optimizer)
            val_loss = trainer.run_epoch(False, val_loader, model, criterion)

            scheduler.step(val_loss)

        # ---- compute metrics at the end ----        
        val_metrics = trainer.evaluate(val_loader, model)
        
        trial.set_user_attr("metrics", val_metrics)

        return val_loss








