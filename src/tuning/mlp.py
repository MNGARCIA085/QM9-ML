import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import ray
from .base import BaseTuner
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.models.nnet import NNModel
from src.utils.metrics import compute_metrics
from src.utils.results import Results, Metrics
from .callbacks import EarlyStopping,LRReducer




class NNTuner(BaseTuner):
    def __init__(self, train_ds, val_ds):
        super().__init__(train_ds, val_ds)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


     # --- Data loaders ---
    def create_loaders(self, batch_size):
        #train_ds = ray.get(self.X_train_id)
        #val_ds = ray.get(self.y_train_id)
        train_ds = self.train_ds
        val_ds = self.val_ds

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        return train_loader, val_loader



    # --- Training one epoch ---
    def run_epoch(loader, model, criterion, optimizer=None):
        model.train() if optimizer else model.eval()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.view(-1)
            target = batch.y.view(-1)
            loss = criterion(pred, target)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    

    # --- Ray train function ---
    def _train_model_ray(self, config):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # small subset for speed
        subset = config["subset"]

        train_loader, val_loader = loaders(
            batch_size=config["batch_size"],
            subset=subset
        )

        model = SimpleMLP(
            num_atom_types=100,
            hidden=config["hidden"]
        ).to(device)

        optimizer = Adam(model.parameters(), lr=config["lr"])
        criterion = MSELoss()

        for epoch in range(config["epochs"]):
            train_loss = run_epoch(train_loader, model, criterion, optimizer)
            val_loss = run_epoch(val_loader, model, criterion)

            # report to Ray Tune
            tune.report({"val_loss":val_loss})




    # Tuning config
    def get_tune_config(self):
        return {
            "hidden": tune.choice([32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32]),
            "epochs": 3,
            "subset": 10,        # only 10 samples → very fast tuning
        }










