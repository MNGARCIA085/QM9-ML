import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.training.gcn import GCNTrainer


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


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_create_model_from_params():
    trainer = GCNTrainer(device="cpu")
    model = trainer.create_model_from_params({"hidden": 16})
    assert isinstance(model, nn.Module)


def test_get_predictions():
    trainer = GCNTrainer(device="cpu")
    loader = DataLoader(make_dataset(4), batch_size=1, shuffle=False)

    model = MockGCN(hidden=16)

    trues, preds = trainer.get_predictions(loader, model)

    assert trues.shape == preds.shape == torch.Size([4])
    assert trues.ndim == preds.ndim == 1


def test_predict():
    trainer = GCNTrainer(device="cpu")
    loader = DataLoader(make_dataset(3), batch_size=1, shuffle=False)

    model = MockGCN(hidden=16)

    preds = trainer.predict(loader, model)

    assert preds.shape == torch.Size([3])
    assert preds.ndim == 1


def test_run_epoch_train():
    trainer = GCNTrainer(device="cpu")
    loader = DataLoader(make_dataset(5), batch_size=1, shuffle=False)

    model = MockGCN(hidden=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = trainer.run_epoch(
        train=True,
        loader=loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_run_epoch_eval():
    trainer = GCNTrainer(device="cpu")
    loader = DataLoader(make_dataset(5), batch_size=1, shuffle=False)

    model = MockGCN(hidden=16)
    criterion = nn.MSELoss()

    loss = trainer.run_epoch(
        train=False,
        loader=loader,
        model=model,
        criterion=criterion,
    )

    assert isinstance(loss, float)
    assert loss > 0
