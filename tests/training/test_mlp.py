import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.training.mlp import MLPTrainer


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


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_create_model_from_params():
    trainer = MLPTrainer(device="cpu")
    model = trainer.create_model_from_params({"hidden": 8})
    assert isinstance(model, nn.Module)


def test_get_predictions():
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(make_dataset(4), batch_size=1, shuffle=False)
    model = MockMLP(hidden=8)

    trues, preds = trainer.get_predictions(loader, model)

    assert trues.shape == preds.shape == torch.Size([4])
    assert trues.ndim == preds.ndim == 1


def test_predict():
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(make_dataset(3), batch_size=1, shuffle=False)
    model = MockMLP(hidden=8)

    preds = trainer.predict(loader, model)

    assert preds.shape == torch.Size([3])
    assert preds.ndim == 1


def test_run_epoch_train():
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(make_dataset(5), batch_size=1, shuffle=False)

    model = MockMLP(hidden=8)
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
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(make_dataset(5), batch_size=1, shuffle=False)

    model = MockMLP(hidden=8)
    criterion = nn.MSELoss()

    loss = trainer.run_epoch(
        train=False,
        loader=loader,
        model=model,
        criterion=criterion,
    )

    assert isinstance(loss, float)
    assert loss > 0
