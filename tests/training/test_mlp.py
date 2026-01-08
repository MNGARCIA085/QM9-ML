import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from qm9_ml.training.mlp import MLPTrainer



def test_create_model_from_params():
    trainer = MLPTrainer(device="cpu")
    model = trainer.create_model_from_params({"hidden": 8})
    assert isinstance(model, nn.Module)


def test_run_epoch_train(dataset_mlp, mock_mlp):
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(dataset_mlp(5), batch_size=1, shuffle=False)

    model = mock_mlp
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = trainer.train_epoch(
        loader=loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_run_epoch_eval(dataset_mlp, mock_mlp):
    trainer = MLPTrainer(device="cpu")
    loader = DataLoader(dataset_mlp(5), batch_size=1, shuffle=False)

    model = mock_mlp
    criterion = nn.MSELoss()

    loss = trainer.val_epoch(
        loader=loader,
        model=model,
        criterion=criterion,
    )

    assert isinstance(loss, float)
    assert loss > 0
