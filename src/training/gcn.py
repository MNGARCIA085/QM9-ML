from .base import BaseTrainer
import torch
import torch.optim as optim
import torch.nn as nn
from src.models.schnet import SchNetRegressor
from src.utils.metrics import compute_metrics
from .registry import TrainerRegistry


from src.models.gcn import SimpleGCN


@TrainerRegistry.register("gcn")
class GCNTrainer(BaseTrainer):
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, epochs=10, device=None, **kwargs):
        super().__init__(train_ds, val_ds, test_ds, epochs=epochs, device=device)



    # create model
    def create_model_from_params(self, params):
        return SimpleGCN(hidden=params["hidden"]).to(self.device)

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

                out = model(batch)          # [num_graphs, 1]
                preds.append(out.view(-1).cpu())

                y = batch.y.view(-1).cpu()  # [num_graphs]
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        return trues, preds


    # run epoch
    def run_epoch(self, train, loader, model, criterion, optimizer=None):
        device = self.device

        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0

        # Use no_grad only for validation
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                batch = batch.to(device)

                out = model(batch)
                pred = out.squeeze(-1)
                target = batch.y.squeeze(-1)

                loss = criterion(pred, target)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)
