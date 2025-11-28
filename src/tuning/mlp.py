from .base import BaseTuner
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.mlp import SimpleMLP
from .registry import TuningRegistry
from src.utils.metrics import compute_metrics



from src.training.mlp import MLPTrainer





@TuningRegistry.register("mlp")
class MLPTuner(BaseTuner):


    trainer_cls = MLPTrainer

    def __init__(self, train_ds, val_ds, epochs=10, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs=epochs, epochs_trials=epochs_trials, device=device)

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
        lr_low = kwargs.get("lr_low", 1e-4)
        lr_high = kwargs.get("lr_high", 1e-2)


        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)


        trainer = self.trainer_cls(self.train_ds, self.val_ds)

        train_loader, val_loader = trainer.create_loaders(batch_size)



        model = self.create_model(trial, hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()


        for _ in range(self.epochs_trials): 
            trainer.run_epoch(True, train_loader, model, criterion, optimizer)
        

        # final validation loss
        val_loss = trainer.run_epoch(False, val_loader, model, criterion)

        # ---- compute additional metrics ----        
        y_true, y_pred = trainer.get_predictions(val_loader, model)


        # metrics
        metrics = compute_metrics(y_true, y_pred)

        # ---- store metadata in the trial ---- (later a dataclass maybe)
        trial.set_user_attr("metrics", metrics)

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
