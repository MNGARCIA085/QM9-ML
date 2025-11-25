from .base import BaseTuner
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.metrics import compute_metrics
from .registry import TuningRegistry
from src.models.gcn import SimpleGCN


@TuningRegistry.register("gcn")
class GCNTuner(BaseTuner):
    def __init__(self, train_ds, val_ds, epochs=10, epochs_trials=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs=epochs, epochs_trials=epochs_trials, device=device)
    
    # create model
    def create_model_from_params(self, params):
        return SimpleGCN(hidden=params["hidden"]).to(self.device)


    # see latr if it is common
    def run_epoch(self, train, loader, model, criterion, optimizer=None): # maybe mode instead of train
        device = self.device
        model.train() if optimizer else model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.squeeze(-1)
            target = batch.y.squeeze(-1)
            loss = criterion(pred, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)


    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    def create_model(self, trial, hidden_opts):
        hidden = trial.suggest_categorical("hidden", hidden_opts)
        # later -> num_atom_types
        return SimpleGCN(hidden=hidden).to(self.device)


    # see later if its not common to all classes
    def objective(self, trial, batch_size_opts=[16, 32], hidden_opts=[32, 64, 128], lr_low=1e-4, lr_high=1e-2):
        batch_size = trial.suggest_categorical("batch_size", batch_size_opts)
        lr = trial.suggest_loguniform("lr", lr_low, lr_high)

        train_loader, val_loader = self.create_loaders(batch_size)

        model = self.create_model(trial, hidden_opts=hidden_opts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs_trials):
            self.run_epoch(True, train_loader, model, criterion, optimizer)
        val_loss = self.run_epoch(False, val_loader, model, criterion)
        

        # ---- compute additional metrics ----
        y_true, y_pred = self.get_predictions(val_loader, model)

        # metrics
        metrics = compute_metrics(y_true, y_pred)

        # ---- store metadata in the trial ---- (later a dataclass maybe)
        trial.set_user_attr("metrics", metrics)


        return val_loss


    # preds
    def get_predictions(self, loader, model):
        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                out = model(batch)          # [num_graphs, 1]
                preds.append(out.view(-1).cpu())

                y = batch.y.view(-1).cpu()  # [num_graphs]
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        return trues, preds

