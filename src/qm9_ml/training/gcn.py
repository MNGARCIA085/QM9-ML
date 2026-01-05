from .base import BaseTrainer
import torch
import torch.optim as optim
import torch.nn as nn
from qm9_ml.models.schnet import SchNetRegressor
from qm9_ml.utils.metrics import compute_metrics
from .registry import TrainerRegistry
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qm9_ml.models.gcn import SimpleGCN


@TrainerRegistry.register("gcn")
class GCNTrainer(BaseTrainer):
    def __init__(self, train_ds=None, val_ds=None, test_ds=None, epochs=10, device=None, **kwargs):
        super().__init__(train_ds, val_ds, test_ds, epochs=epochs, device=device)

    # create model
    def create_model_from_params(self, params):
        return SimpleGCN(hidden=params["hidden"]).to(self.device)

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
                
                #pred = out.squeeze(-1)
                #target = batch.y.squeeze(-1)
                pred = out.view(-1)
                target = batch.y.view(-1)


                loss = criterion(pred, target)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)
