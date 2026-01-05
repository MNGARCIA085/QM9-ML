from .base import BasePredictor
import torch
from .registry import PredictorRegistry
from torch_geometric.loader import DataLoader


@PredictorRegistry.register("mlp")
class MLPPredictor(BasePredictor):


    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    def predict_with_targets(self, loader):
        preds = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                y_hat = self.model(batch)            # shape [batch_size]
                y = batch.y.view(-1).to(self.device) # ensure [batch_size]

                preds.append(y_hat.cpu())
                trues.append(y.cpu())

        preds = torch.cat(preds)
        trues = torch.cat(trues)
        return trues, preds

    # for inference
    def predict(self, data):
        """Run inference and return only predictions."""

        # --- Normalize input ---
        loader = self._normalize_loader(data)


        # --- Preds ----
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                y_hat = self.model(batch)  # shape [batch_size]
                y = batch.y.view(-1).to(self.device)
                preds.append(y_hat.cpu())

        return torch.cat(preds)


# important for indent -> sed -i 's/\t/    /g' base.py
