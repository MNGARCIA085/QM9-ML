import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna

import mlflow
import os
import tempfile


from qm9_ml.utils.metrics import compute_metrics
from .callbacks.early_stopping import EarlyStopping
from .callbacks.checkpoint import ModelCheckpoint
from .callbacks.lr_schedulers import get_plateau_scheduler




class BaseTrainer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        self.patience = 30 # for early stopping



    # create model
    def create_model_from_params(self, params):
        raise NotImplementedError

    # loaders
    def create_loaders(self, train_ds, val_ds, batch_size):
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


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
        return get_plateau_scheduler(
            optimizer,
            mode="min",
            factor = params.get("factor", 0.7),
            patience = params.get("patience", 2),
            min_lr = params.get("min_lr", 1e-6)
        )


    # ------------------------------------------------------------#

    def _step(self, batch, model, criterion):
        """ Model-specific forward + loss logic """
        raise NotImplementedError("Subclasses must implement it")



    def train_epoch(self, loader, model, criterion, optimizer):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            loss = self._step(batch, model, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    def val_epoch(self, loader, model, criterion):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                loss = self._step(batch, model, criterion)
                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)



    # train
    def train(self, params, train_ds, val_ds, epochs=10): 

        # params: schnet ex:{'batch_size': 16, 'lr': .., 'hidden_channels': 256, 'num_filters': 128, 'num_interactions': 4}
        # those the ones I use for tuning, but i also have some fixed (limited resources) like cutoff

        """
        Train a fresh model using the given params.
        """

        # Loaders
        train_loader, val_loader = self.create_loaders(train_ds, val_ds, params["batch_size"])

        # Rebuild best model (subclasses must implement create_model_from_params)
        model = self.create_model_from_params(params)


        # can vary depending on the model
        optimizer = self.configure_optimizer(model, params)
        criterion = nn.MSELoss()

        # ---- scheduler----
        scheduler = self.configure_scheduler(optimizer, params)


        # ---- early stopping and model checkpoint  ----
        early_stop = EarlyStopping(patience=self.patience, mode="min")
        
        #ckpt_path = tempfile.mktemp(suffix=".pt") # creates something like /tmp/tmpabcd1234.pt, goal: avoid collisions
        ckpt_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name #; not used mktemp (race conditions)
        ckpt = ModelCheckpoint(ckpt_path, mode="min")


        train_losses = []
        val_losses = []

        final_epoch = 0          # default, will update
        final_lr = None


        for epoch in range(epochs):

            # store epoch (useful for ex. to know which was my best epoch)
            self.current_epoch = epoch

            train_loss = self.train_epoch(train_loader, model, criterion, optimizer)
            val_loss = self.val_epoch(val_loader, model, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # if i want a history per metric I can write a fn. eval_one_epcoch and calculate metrics
            print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")


            # ---- 1) LR Scheduler ----
            scheduler.step(val_loss)
            final_lr = optimizer.param_groups[0]["lr"]  # always update
            """
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR after scheduler: {current_lr:.6f}")
            """

            # ---- 2) Checkpoint ----
            is_best = ckpt.step(val_loss, model)
            if is_best:
                print(f"[BEST MODEL] Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
                print("Checkpoint updated")


            # ---- 3) Early stopping ----
            early_stop.step(val_loss)
            if early_stop.stop_training:
                print("Early stopping triggered!")
                final_epoch = epoch + 1  # human-readable
                break

            final_epoch = epoch + 1  # update each loop


        # Load best model
        model.load_state_dict(torch.load(ckpt_path))

        # Remove temporary checkpoint
        os.remove(ckpt_path)
        
        # all hyperparams
        hyperparams = {
            **model.config,     # unpack existing config
            "epochs": epochs,
            "final_epochs": final_epoch,
            "final_lr": final_lr,
            "lr": params["lr"],
            "batch_size": params["batch_size"],
        }


        # --- Return results ---
        return(
                {
                    "model": model,
                    "train":
                        {
                            "losses": train_losses,
                        },
                    "val":{
                        "losses": val_losses,
                    },
                    "hyperparams": hyperparams,
                }
            )



