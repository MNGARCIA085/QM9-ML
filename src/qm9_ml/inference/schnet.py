from .base import BasePredictor
import torch
from .registry import PredictorRegistry





@PredictorRegistry.register("schnet")
class SchNetPredictor(BasePredictor):


    # returns preds AND true labels
    def predict_with_targets(self, loader):

        preds, trues = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                y_hat = self.model(batch.z, batch.pos, batch.batch).squeeze(-1)
                y = batch.y.squeeze(-1).float()

                preds.append(y_hat.cpu())
                trues.append(y.cpu())

        trues = torch.cat(trues)
        preds = torch.cat(preds)
        return trues, preds



    # for inference
    def predict(self, data):
        """Run inference and return only predictions."""


        # --- Normalize input ---
        loader = self._normalize_loader(data)
        
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                y_hat = self.model(batch.z, batch.pos, batch.batch).squeeze(-1)
                preds.append(y_hat.cpu())

        return torch.cat(preds)










