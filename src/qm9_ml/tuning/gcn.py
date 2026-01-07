from .base import BaseTuner
import torch
import torch.nn as nn
import torch.optim as optim
from .registry import TuningRegistry
from qm9_ml.models.gcn import SimpleGCN
from qm9_ml.training.gcn import GCNTrainer
from qm9_ml.inference.gcn import GCNPredictor
from qm9_ml.utils.metrics import compute_metrics


@TuningRegistry.register("gcn")
class GCNTuner(BaseTuner):

    trainer_cls = GCNTrainer
    predictor_cls = GCNPredictor

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
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)

        # trainer
        trainer = self.trainer_cls(device=self.device) 

        # loaders
        train_loader, val_loader = trainer.create_loaders(self.train_ds, self.val_ds, batch_size)

        # model, optimizer, criterion
        model = self.create_model(trial, hidden_opts=hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # training loop
        for _ in range(self.epochs_trials):
            trainer.train_epoch(train_loader, model, criterion, optimizer)
        val_loss = trainer.val_epoch(val_loader, model, criterion) # only last
        
        # ---- compute metrics ----
        predictor = self.predictor_cls(model=model, device=self.device)
        y_true, y_pred = predictor.predict_with_targets(val_loader)
        val_metrics = compute_metrics(y_true, y_pred)


        # ---- store metadata in the trial ----
        trial.set_user_attr("metrics", val_metrics)

        return val_loss


