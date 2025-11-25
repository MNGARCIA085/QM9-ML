from .base import BaseTuner
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from src.models.mlp import SimpleMLP
import torch


from src.utils.metrics import compute_metrics


class MLPTuner(BaseTuner):
    def __init__(self, train_ds, val_ds, epochs=10, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs=epochs, epochs_trials=epochs_trials, device=device)

        # Any MLPTuner-specific attributes
        #self.hidden_dim = kwargs.get("hidden_dim", 128)
        #...................


    # run one epoch; maybe later a common fn.
    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        model.train() if train else model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1).float())
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)
    


    # get preds
    def get_predictions(self, loader, model):
        model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                y_hat = model(batch)                 # shape [batch_size]
                y = batch.y.view(-1).to(self.device) # ensure [batch_size]

                preds.append(y_hat.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds)
        trues = torch.cat(trues)
        return trues, preds


    # create model
    def create_model_from_params(self, params):
        return SimpleMLP(hidden=params["hidden"]).to(self.device)


    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    def create_model(self, trial, hidden_opts):
        hidden = trial.suggest_categorical("hidden", hidden_opts)
        return SimpleMLP(hidden=hidden).to(self.device)




    def objective(self, trial, batch_size_opts=[16, 32], hidden_opts=[32, 64, 128], lr_low=1e-4, lr_high=1e-2):
        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)

        train_loader, val_loader = self.create_loaders(batch_size)

        model = self.create_model(trial, hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs_trials): 
            self.run_epoch(True, train_loader, model, criterion, optimizer)
        

        # final validation loss
        val_loss = self.run_epoch(False, val_loader, model, criterion)

        # ---- compute additional metrics ----
        y_true, y_pred = self.get_predictions(val_loader, model)


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
