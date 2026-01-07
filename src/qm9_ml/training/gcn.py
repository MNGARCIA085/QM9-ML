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
    def __init__(self, epochs=10, device=None, **kwargs):
        super().__init__(epochs=epochs, device=device)


    # create model
    def create_model_from_params(self, params):
        return SimpleGCN(hidden=params["hidden"]).to(self.device)


    # ------ Step ------
    def _step(self, batch, model, criterion):
        out = model(batch)
        #pred = out.squeeze(-1)
        #target = batch.y.squeeze(-1)
        pred = out.view(-1)
        target = batch.y.view(-1)
        return criterion(pred, target)
