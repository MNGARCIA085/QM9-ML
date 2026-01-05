from .base import BasePredictor
import torch
from .registry import PredictorRegistry



@PredictorRegistry.register("gcn")
class GCNPredictor(BasePredictor):

    def predict_with_targets(self, loader):

        preds = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch)          # [num_graphs, 1]
                preds.append(out.view(-1).cpu())

                y = batch.y.view(-1).cpu()  # [num_graphs]
                trues.append(y)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        return trues, preds


    # preds
    def predict(self, data):
        """Run inference and return predictions."""

        # --- Normalize input ---
        loader = self._normalize_loader(data)
        
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)          # [num_graphs, 1]
                preds.append(out.view(-1).cpu())
                y = batch.y.view(-1).cpu()  # [num_graphs]

        preds = torch.cat(preds)
        return preds



