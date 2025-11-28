from .base import BaseTrainer
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.schnet import SchNetRegressor
from src.utils.metrics import compute_metrics
from .registry import TrainerRegistry


from src.models.mlp import SimpleMLP


@TrainerRegistry.register("mlp")
class MLPTrainer(BaseTrainer):
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, epochs=10, device=None, **kwargs):
        super().__init__(train_ds, val_ds, test_ds, epochs=epochs, device=device)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
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



    # run epoch
    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        device = self.device

        if train:
            model.train()
            context = torch.enable_grad()
        else:
            model.eval()
            context = torch.no_grad()

        total_loss = 0

        with context:
            for batch in loader:
                batch = batch.to(device)

                out = model(batch)
                loss = criterion(out.view(-1), batch.y.view(-1).float())

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    # create model
    def create_model_from_params(self, params):
        return SimpleMLP(hidden=params["hidden"]).to(self.device)

