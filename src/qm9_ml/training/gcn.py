from .registry import TrainerRegistry
from .base import BaseTrainer
from qm9_ml.models.gcn import SimpleGCN


@TrainerRegistry.register("gcn")
class GCNTrainer(BaseTrainer):
    def __init__(self, device=None, **kwargs):
        super().__init__(device=device)


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
