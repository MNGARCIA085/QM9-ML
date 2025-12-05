import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.training.schnet import SchNetTrainer


# ------------------------------------------------------------
# Helper: create tiny synthetic SchNet-style dataset
# ------------------------------------------------------------
def make_synthetic_schnet_dataset(num_samples=20, num_nodes=5):
    dataset = []
    for _ in range(num_samples):
        pos = torch.randn(num_nodes, 3)          # Positions
        z = torch.randint(1, 10, (num_nodes,))   # Atomic numbers
        y = torch.randn(1)                       # Scalar property

        # SchNet does NOT require edge_index explicitly
        data = Data(pos=pos, z=z, y=y, num_nodes=num_nodes)
        dataset.append(data)

    return dataset


# ------------------------------------------------------------
# Test: run_epoch
# ------------------------------------------------------------
def test_schnet_run_epoch():
    train_ds = make_synthetic_schnet_dataset(10, 6)
    train_loader = DataLoader(train_ds, batch_size=4)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer(train_ds=train_ds, val_ds=None, test_ds=None, epochs=1)
    model = trainer.create_model_from_params(params)

    optimizer = trainer.configure_optimizer(model, params)
    criterion = torch.nn.MSELoss()

    loss = trainer.run_epoch(True, train_loader, model, criterion, optimizer)

    assert isinstance(loss, float)
    assert loss >= 0.0


# ------------------------------------------------------------
# Test: get_predictions
# ------------------------------------------------------------
def test_schnet_get_predictions():
    ds = make_synthetic_schnet_dataset(12, 5)
    loader = DataLoader(ds, batch_size=4)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer(train_ds=None, val_ds=None, test_ds=None)
    model = trainer.create_model_from_params(params)

    y_true, y_pred = trainer.get_predictions(loader, model)

    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 1  # (num_samples,)
    assert len(y_true) == len(ds)


# ------------------------------------------------------------
# Test: predict
# ------------------------------------------------------------
def test_schnet_predict():
    ds = make_synthetic_schnet_dataset(12, 5)
    loader = DataLoader(ds, batch_size=4)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer(train_ds=None, val_ds=None)
    model = trainer.create_model_from_params(params)

    preds = trainer.predict(loader, model)

    assert preds.dim() == 1
    assert len(preds) == len(ds)


# ------------------------------------------------------------
# Test: full training loop (train_best_model)
# ------------------------------------------------------------
def test_schnet_train_best_model_runs():
    train_ds = make_synthetic_schnet_dataset(20, 6)
    val_ds = make_synthetic_schnet_dataset(10, 6)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer(train_ds=train_ds, val_ds=val_ds, epochs=3)
    result = trainer.train(params)

    # model returned
    assert "model" in result

    # losses recorded
    assert len(result["train"]["losses"]) <= 3
    assert len(result["val"]["losses"]) <= 3

    # metrics exist
    assert "metrics" in result["train"]
    assert "metrics" in result["val"]

    # hyperparams returned
    assert result["hyperparams"] == params




# ------------------------------------------------------------
# Configure optimizer
# ------------------------------------------------------------
"""
Check:
    Returned object is torch.optim.AdamW
    LR is correct
    Weight decay is correct
"""
def test_configure_optimizer():
    trainer = SchNetTrainer(device="cpu")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)

    model = DummyModel()

    params = {"lr": 1e-3, "weight_decay": 5e-4}
    opt = trainer.configure_optimizer(model, params)

    assert isinstance(opt, torch.optim.AdamW)
    for group in opt.param_groups:
        assert group["lr"] == 1e-3
        assert group["weight_decay"] == 5e-4






"""
✔ They test the real SchNetTrainer (not a dummy)

real forward pass

real gradients

real DataLoader behavior

real SchNetRegressor

✔ They run extremely fast

tiny datasets

tiny model

only a few epochs

✔ They validate behavior, not numerical accuracy

ensures your training loop does not break

ensures model training flows end-to-end

ensures outputs have correct shapes

ensures all trainer hooks work

✔ They remain robust

Because we construct synthetic, valid SchNet data instead of depending on your full preprocessing pipeline.
"""



