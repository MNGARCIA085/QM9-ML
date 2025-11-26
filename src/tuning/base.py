# ---------------------------------------------------------
# Base tuner
# ---------------------------------------------------------
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
from torch_geometric.loader import DataLoader


class BaseTuner:
    def __init__(self, train_ds, val_ds, epochs=10, epochs_trials=5, device=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs  
        self.epochs_trials = epochs_trials # small epochs for quick test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None


    # loaders
    def create_loaders(self, batch_size):
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


    # not all run epchos are the sme, have that in count
    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        raise NotImplementedError("Subclasses must implement it")


    def objective(self, trial, **kwargs):
        raise NotImplementedError("Subclasses must implement objective()")

    

    #
    def tune(self, n_trials=10, **kwargs):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, **kwargs), # to pass kwargs
                       n_trials=n_trials)


        print("Best params:", study.best_params)
        best_attrs = study.best_trial.user_attrs # metrics are a key under this
        print("All metrics:", best_attrs)

        

        # store for later (MLflow, logging, checkpoints…)
        self.best_params = study.best_params
        self.best_attrs  = best_attrs   # metrics / extras (right now i just for my best trial for a few epochs)


        # ---- train best model ----
        results = self.train_best_model(self.best_params)
        # return more...........



        return results, study.best_params, best_attrs
    



    def train_best_model(self, best_params):
        """
        Train a fresh model using the best hyperparameters.
        """

        # Loaders
        train_loader, val_loader = self.create_loaders(best_params["batch_size"])

        # Rebuild best model (subclasses must implement create_model_from_params)
        model = self.create_model_from_params(best_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            train_loss = self.run_epoch(True, train_loader, model, criterion, optimizer)
            val_loss   = self.run_epoch(False, val_loader, model, criterion)
            print(f"[BEST MODEL] Epoch {epoch+1}/{self.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")


        # ---- compute additional metrics ----
        y_true, y_pred = self.get_predictions(val_loader, model)

        # metrics (compute again; for my best model for trained for a larger period of time)
        from src.utils.metrics import compute_metrics
        metrics = compute_metrics(y_true, y_pred)


    
        # return
        return(
                {
                    "model": model,
                    "train_loss": train_loss,
                    "val_loss": val_loss, # later history
                    "val_metrics": metrics,
                    "hyperparams": best_params,
                }
            )

        #return model


    def get_preds(self, loader, model):
        """Child must implement. Returns y_true, y_pred"""
        raise NotImplementedError


    



