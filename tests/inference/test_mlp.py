import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qm9_ml.inference.mlp import MLPPredictor


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_dataset(num_graphs=5, hidden=8):
    """
    Creates a dataset of simple Data objects for the MLP.
    Each graph has x as a single graph-level feature vector.
    """
    dataset = []
    for _ in range(num_graphs):
        x = torch.rand(1, hidden)             # MLP takes graph-level features
        y = torch.rand(1)                     # regression target
        dataset.append(Data(x=x, y=y))
    return dataset


# A simple mock MLP: mean over features → linear layer → scalar output
class MockMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        # data.x: [1, hidden]
        return self.lin(data.x).view(-1)  # returns [1]



# ---------------------------------------------------------------
def test_get_predictions():
    loader = DataLoader(make_dataset(4), batch_size=1, shuffle=False)
    model = MockMLP(hidden=8)
    predictor = MLPPredictor(model=model, device="cpu")

    trues, preds = predictor.predict_with_targets(loader)

    assert trues.shape == preds.shape == torch.Size([4])
    assert trues.ndim == preds.ndim == 1


def test_predict():
    loader = DataLoader(make_dataset(3), batch_size=1, shuffle=False)
    model = MockMLP(hidden=8)
    predictor = MLPPredictor(model=model, device="cpu")

    preds = predictor.predict(loader)

    assert preds.shape == torch.Size([3])
    assert preds.ndim == 1


