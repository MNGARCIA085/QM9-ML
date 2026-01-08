import torch
from torch_geometric.loader import DataLoader
from qm9_ml.training.schnet import SchNetTrainer



# ------------------------------------------------------------
# Test: run_epoch
# ------------------------------------------------------------
def test_schnet_train_epoch(dataset_schnet):
    train_ds = dataset_schnet(10, 6)
    train_loader = DataLoader(train_ds, batch_size=4)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer()
    model = trainer.create_model_from_params(params)

    optimizer = trainer.configure_optimizer(model, params)
    criterion = torch.nn.MSELoss()

    loss = trainer.train_epoch(train_loader, model, criterion, optimizer)

    assert isinstance(loss, float)
    assert loss >= 0.0




# ------------------------------------------------------------
# Test: full training loop (train_best_model)
# ------------------------------------------------------------
def test_schnet_train_best_model_runs(dataset_schnet):
    train_ds = dataset_schnet(20, 6)
    val_ds = dataset_schnet(10, 6)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer()
    result = trainer.train(params, train_ds=train_ds, val_ds=val_ds, epochs=3)

    # model returned
    assert "model" in result

    # losses recorded
    assert len(result["train"]["losses"]) <= 3
    assert len(result["val"]["losses"]) <= 3

    # hyperparams returned
    assert "epochs" in result["hyperparams"]




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






