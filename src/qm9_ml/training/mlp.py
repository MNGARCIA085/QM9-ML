from .base import BaseTrainer
import torch
import torch.optim as optim
from .registry import TrainerRegistry
from qm9_ml.models.mlp import SimpleMLP


@TrainerRegistry.register("mlp")
class MLPTrainer(BaseTrainer):
    def __init__(self, epochs=10, device=None, **kwargs):
        super().__init__(epochs=epochs, device=device)


    # create model
    def create_model_from_params(self, params):
        return SimpleMLP(hidden=params["hidden"]).to(self.device)


    # ------ Step  ------
    def _step(self, batch, model, criterion):
        out = model(batch)
        return criterion(out.view(-1), batch.y.view(-1).float())


