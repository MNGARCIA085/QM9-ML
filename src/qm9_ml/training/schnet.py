from .base import BaseTrainer
import torch
import torch.optim as optim
from .registry import TrainerRegistry
from qm9_ml.models.schnet import SchNetRegressor
from qm9_ml.utils.metrics import compute_metrics



@TrainerRegistry.register("schnet")
class SchNetTrainer(BaseTrainer):
    def __init__(self, epochs=10, device=None, **kwargs):
        super().__init__(epochs=epochs, device=device)


    # create model
    def create_model_from_params(self, params):
        return SchNetRegressor(
            hidden_channels=params["hidden_channels"],
            num_filters=params["num_filters"],
            num_interactions=params["num_interactions"]
        ).to(self.device)  # later cutoff


    # override optimizer
    def configure_optimizer(self, model, params):
        return torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 1e-5),
        )


    # ------ Step  ------
    def _step(self, batch, model, criterion):
        out = model(batch.z, batch.pos, batch.batch)
        pred = out.squeeze(-1)
        target = batch.y.squeeze(-1)
        return criterion(pred, target)

