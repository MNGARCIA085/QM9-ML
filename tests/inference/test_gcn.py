import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qm9_ml.inference.gcn import GCNPredictor


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_dataset(num_graphs=5, hidden=16):
    """
    Creates a list of real PyG Data objects so DataLoader works correctly.
    """
    dataset = []
    for _ in range(num_graphs):
        x = torch.rand(10, hidden)               # node features
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # trivial edges
        y = torch.rand(1)                        # graph-level target
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


# Mock model (same idea as SimpleGCN, but extremely minimal)
class MockGCN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        # simple graph-level prediction = mean over node features
        out = self.lin(data.x).mean(dim=0, keepdim=True)  # [1,1]
        return out



#-------------------------------------------------------------------------------#


def test_get_predictions():
    loader = DataLoader(make_dataset(4), batch_size=1, shuffle=False)
    model = MockGCN(hidden=16)
    
    predictor = GCNPredictor(model=model, device="cpu")
    

    trues, preds = predictor.predict_with_targets(loader)

    assert trues.shape == preds.shape == torch.Size([4])
    assert trues.ndim == preds.ndim == 1


def test_predict():
    loader = DataLoader(make_dataset(3), batch_size=1, shuffle=False)
    model = MockGCN(hidden=16)

    predictor = GCNPredictor(model=model, device="cpu")
    
    preds = predictor.predict(loader)

    assert preds.shape == torch.Size([3])
    assert preds.ndim == 1