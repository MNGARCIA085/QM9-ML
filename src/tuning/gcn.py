from .base import BaseTuner
import torch
import torch.nn as nn
import torch.optim as optim
from .registry import TuningRegistry
from src.models.gcn import SimpleGCN
from src.training.gcn import GCNTrainer


@TuningRegistry.register("gcn")
class GCNTuner(BaseTuner):

    trainer_cls = GCNTrainer

    def __init__(self, train_ds, val_ds, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs_trials=epochs_trials, device=device)
    
    # create model
    def create_model_from_params(self, params):
        return SimpleGCN(hidden=params["hidden"]).to(self.device)

    
    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    def create_model(self, trial, hidden_opts):
        hidden = trial.suggest_categorical("hidden", hidden_opts)
        return SimpleGCN(hidden=hidden).to(self.device)


    # see later if its not common to all classes
    def objective(self, trial, **kwargs):

        # specific params
        batch_size_opts = kwargs.get('batch_size_opts', [16, 32])
        hidden_opts  = kwargs.get('hidden_opts', [32, 64, 128])
        lr_low = kwargs.get("lr", {}).get("low", 1e-4)
        lr_high = kwargs.get("lr", {}).get("high", 1e-2)

        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)

        # trainer
        trainer = self.trainer_cls(self.train_ds, self.val_ds) 

        # loaders
        train_loader, val_loader = trainer.create_loaders(batch_size)

        # model, optimizer, criterion
        model = self.create_model(trial, hidden_opts=hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # training loop
        for _ in range(self.epochs_trials):
            trainer.run_epoch(True, train_loader, model, criterion, optimizer)
        val_loss = trainer.run_epoch(False, val_loader, model, criterion) # only last
        
        # ---- compute metrics ----
        val_metrics = trainer.evaluate(val_loader, model)

        # ---- store metadata in the trial ----
        trial.set_user_attr("metrics", val_metrics)

        return val_loss


