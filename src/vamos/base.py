# base_tuner.py
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna

class BaseTuner:
    def __init__(self, train_ds, val_ds, device=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run_epoch(self, loader, model, criterion, optimizer=None):
        model.train() if optimizer else model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1).float())
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    def objective(self, trial):
        raise NotImplementedError("Subclasses must implement objective()")

    def tune(self, n_trials=10):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        return study.best_params
