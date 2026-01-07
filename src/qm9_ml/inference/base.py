from torch_geometric.data import Data
from torch_geometric.loader import DataLoader





# a preictor, always the same model if i pass it in init

class BasePredictor:

    def __init__(self, model, batch_size=32, device="cpu"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size


    # normalize loader
    def _normalize_loader(self, data):
        if isinstance(data, DataLoader):
            loader = data

        elif isinstance(data, Data):
            loader = DataLoader([data], batch_size=1, shuffle=False)

        else:
            # Assume iterable of Data
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)

        return loader


    # get preds and labels
    def predict_with_targets(self, loader):
        """Child must implement. Returns y_true, y_pred"""
        raise NotImplementedError

    # predict
    def predict(self, data):
        """Child must implement. Returns y_pred"""
        raise NotImplementedError


    