import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna

import mlflow
import os
import tempfile


from src.utils.metrics import compute_metrics
from .callbacks.early_stopping import EarlyStopping
from .callbacks.checkpoint import ModelCheckpoint
from .callbacks.lr_schedulers import get_plateau_scheduler




class BaseTrainer:
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, epochs=10, device=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.epochs = epochs  
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        self.patience = 30 # for early stopping



    # get preds and labels
    def get_predictions(self, loader, model):
        """Child must implement. Returns y_true, y_pred"""
        raise NotImplementedError

    # predict
    def predict(self, loader_or_data, model, batch_size):
        """Child must implement. Returns y_pred"""
        raise NotImplementedError

    # evaluate a given loader
    def evaluate(self, loader, model, batch_size=32):
        """Evaluate the model on a dataset or DataLoader."""
        
        # If it's not already a DataLoader, wrap it
        if not isinstance(loader, DataLoader):
            loader = DataLoader(loader, batch_size=batch_size, shuffle=False)

        y_true, y_pred = self.get_predictions(loader, model)
        return compute_metrics(y_true, y_pred)

    
    def run_epoch(self, train, loader, model, criterion, optimizer):
        raise NotImplementedError("Subclasses must implement it")


    def create_model_from_params(self, params):
        raise NotImplementedError




    #-------------------------------------------------------------------------

    # loaders
    def create_loaders(self, batch_size):
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader



    #-------------------------------------------------------------------------

    # conf. optimizer, I can override it in the subclasses if i need it
    def configure_optimizer(self, model, params):
        return torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 1e-6),
        )


    # subclasses can override it
    def configure_scheduler(self, optimizer, params):
        # default scheduler (Plateau)
        print(params)
        return get_plateau_scheduler(
            optimizer,
            mode="min",
            factor = params.get("factor", 0.7),
            patience = params.get("patience", 2),
            min_lr = params.get("min_lr", 1e-6)
        )



    # train
    def train(self, params):


        # params: schnet ex:{'batch_size': 16, 'lr': .., 'hidden_channels': 256, 'num_filters': 128, 'num_interactions': 4}

        """
        Train a fresh model using the best hyperparameters.
        """

        # Loaders
        train_loader, val_loader = self.create_loaders(params["batch_size"])

        # Rebuild best model (subclasses must implement create_model_from_params)
        model = self.create_model_from_params(params)


        # can vary depending on the model
        optimizer = self.configure_optimizer(model, params)
        criterion = nn.MSELoss()

        # ---- scheduler----
        scheduler = self.configure_scheduler(optimizer, params)


        # ---- early stopping and model checkpoint  ----
        early_stop = EarlyStopping(patience=self.patience, mode="min")
        ckpt_path = tempfile.mktemp(suffix=".pt") # creates something like /tmp/tmpabcd1234.pt, goal: avoid collisions
        ckpt = ModelCheckpoint(ckpt_path, mode="min") 


        train_losses = []
        val_losses = []


        for epoch in range(self.epochs):

            # store epoch (useful for ex. to know which was my best epoch)
            self.current_epoch = epoch

            train_loss = self.run_epoch(True, train_loader, model, criterion, optimizer)
            val_loss = self.run_epoch(False, val_loader, model, criterion)


            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # if i want a history per metric I can write a fn. eval_one_epcoch and calculate metrics

            print(f"[BEST MODEL] Epoch {epoch+1}/{self.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")


            # ---- 1) LR Scheduler ----
            scheduler.step(val_loss)
            """
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR after scheduler: {current_lr:.6f}")
            """

            # ---- 2) Checkpoint ----
            is_best = ckpt.step(val_loss, model)
            if is_best:
                print("Checkpoint updated")


            # ---- 3) Early stopping ----
            early_stop.step(val_loss)
            if early_stop.stop_training:
                print("Early stopping triggered!")
                break



        # Load best model
        model.load_state_dict(torch.load(ckpt_path))

        # Remove temporary checkpoint
        os.remove(ckpt_path)
        
        # ---- compute additional metrics (for train and val; final metrics) ----
        train_metrics = self.evaluate(train_loader, model)
        val_metrics = self.evaluate(val_loader, model)


        # --- Return results ---
        return(
                {
                    "model": model,
                    "train":
                        {
                            "losses": train_losses,
                            "metrics": train_metrics,
                        },
                    "val":{
                        "losses": val_losses,
                        "metrics": val_metrics,
                    },
                    "hyperparams": params,
                }
            )


    

# https://chatgpt.com/c/6930c0c8-5764-8329-96d5-72f43a5b31d4
