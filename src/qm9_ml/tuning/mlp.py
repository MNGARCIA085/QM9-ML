from .base import BaseTuner
import torch
import torch.nn as nn
import torch.optim as optim
from .registry import TuningRegistry
from qm9_ml.models.mlp import SimpleMLP
from qm9_ml.training.mlp import MLPTrainer
from qm9_ml.inference.mlp import MLPPredictor
from qm9_ml.utils.metrics import compute_metrics


@TuningRegistry.register("mlp")
class MLPTuner(BaseTuner):


    trainer_cls = MLPTrainer
    predictor_cls = MLPPredictor


    def __init__(self, train_ds, val_ds, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs_trials=epochs_trials, device=device)

        # Any MLPTuner-specific attributes
        #self.hidden_dim = kwargs.get("hidden_dim", 128)
        #...................
    

    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    def create_model(self, trial, hidden_opts):
        hidden = trial.suggest_categorical("hidden", hidden_opts)
        return SimpleMLP(hidden=hidden).to(self.device)


    def objective(self, trial, **kwargs):
        
        # specific params
        batch_size_opts = kwargs.get('batch_size_opts', [16, 32])
        hidden_opts  = kwargs.get('hidden_opts', [32, 64, 128])
        lr_low = kwargs.get("lr", {}).get("low", 1e-4)
        lr_high = kwargs.get("lr", {}).get("high", 1e-2)



        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)


        trainer = self.trainer_cls(device=self.device)

        train_loader, val_loader = trainer.create_loaders(self.train_ds, self.val_ds, batch_size)


        model = self.create_model(trial, hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()


        for _ in range(self.epochs_trials): 
            trainer.train_epoch(train_loader, model, criterion, optimizer)
        

        # final validation loss
        val_loss = trainer.val_epoch(val_loader, model, criterion)
      
        # ---- compute val metrics ----
        predictor = self.predictor_cls(model=model, device=self.device)
        y_true, y_pred = predictor.predict_with_targets(val_loader)
        val_metrics = compute_metrics(y_true, y_pred)


        # ---- store metadata in the trial ---- (later a dataclass maybe)
        trial.set_user_attr("metrics", val_metrics)

        # return
        return val_loss   # Optuna must optimize a scalar






"""

Bonus: generic helper to extract full leaderboard of trials

If you want a table of all trials + all metrics:

def trials_to_dataframe(study):
    rows = []
    for t in study.trials:
        row = {}
        row.update(t.params)
        row.update(t.user_attrs)
        row["value"] = t.value
        row["number"] = t.number
        rows.append(row)
    return rows


"""
