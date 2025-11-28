from .base import BaseTrainer
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.schnet import SchNetRegressor
from src.utils.metrics import compute_metrics
from .registry import TrainerRegistry



@TrainerRegistry.register("schnet")
class SchNetTrainer(BaseTrainer):
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, epochs=10, device=None, **kwargs):
        super().__init__(train_ds, val_ds, test_ds, epochs=epochs, device=device)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    def get_predictions(self, loader, model):
        model.eval()

        preds, trues = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                y_hat = model(batch.z, batch.pos, batch.batch).squeeze(-1)
                y = batch.y.squeeze(-1).float()

                preds.append(y_hat.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds)
        trues = torch.cat(trues)
        return trues, preds


    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0

        # --- IMPORTANT ---
        # Use torch.no_grad() only when NOT training
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                batch = batch.to(self.device)

                if train:
                    optimizer.zero_grad()

                out = model(batch.z, batch.pos, batch.batch)
                pred = out.squeeze(-1)
                target = batch.y.squeeze(-1)

                loss = criterion(pred, target)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)



    # create model
    def create_model_from_params(self, params):
        return SchNetRegressor(
            hidden_channels=params["hidden_channels"],
            num_filters=params["num_filters"],
            num_interactions=params["num_interactions"]
        ).to(self.device)  # later cutoff

