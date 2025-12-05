import os
import torch
import torch.nn as nn
import tempfile
from torch_geometric.data import Data
from src.training.callbacks.checkpoint import ModelCheckpoint
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
# Early Stopping
# ------------------------------------------------------------
"""
Checks
    Early stopping engages at the correct patience
    Best model loss is updated
    Training stops early

-> later: in the docstring!!!

"""
def test_early_stopping(monkeypatch):

    train_ds = make_synthetic_schnet_dataset(20, 6)
    val_ds = make_synthetic_schnet_dataset(10, 6)

    params = {
        "lr": 1e-3,
        "batch_size": 4,
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
    }

    trainer = SchNetTrainer(train_ds=train_ds, val_ds=val_ds, epochs=40)

    def fake_run_epoch(train, loader, model, criterion, optimizer=None):
        #return val_losses.pop(0)
        return 0.123 # dummy data, never improves


    def fake_eval(loader, model):
        return 0.5

    monkeypatch.setattr(
        trainer, "run_epoch", fake_run_epoch, raising=True
    )
    monkeypatch.setattr(
        trainer, "evaluate", fake_eval, raising=True
    )

    trainer.patience = 2

    trainer.train(params)

    # Should not reach patience+1 epochs
    assert trainer.current_epoch < (trainer.patience+1)




# ------------------------------------------------------------
# LR Scheduler
# ------------------------------------------------------------
"""
Checks
    Mode is “min”
    LR decreases after plateau
    Patience is respected
    min_lr is respected
"""
def test_plateau_scheduler_step():
    trainer = SchNetTrainer(device="cpu")

    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = trainer.configure_scheduler(
        optimizer, 
        params={"lr": 1e-3, "patience": 0, "factor": 0.5, "min_lr": 1e-5}
    )

    # First step: no reduction yet
    scheduler.step(1.0)
    lr1 = optimizer.param_groups[0]["lr"]

    # No improvement → should reduce
    scheduler.step(1.0)
    lr2 = optimizer.param_groups[0]["lr"]

    assert lr2 < lr1
    assert lr2 == lr1 * 0.5 or lr2 == 1e-5



# ------------------------------------------------------------
# Model Checkpoint
# ------------------------------------------------------------
def test_checkpoint_saves_on_improvement():
    model = nn.Linear(10, 1)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "best.pt")

        ckpt = ModelCheckpoint(ckpt_path, mode="min")

        # First step: always saves
        improved = ckpt.step(metric=0.5, model=model)
        assert improved is True
        assert os.path.exists(ckpt_path)
        first_best = ckpt.best

        # Worse metric: should NOT save
        improved = ckpt.step(metric=0.8, model=model)
        assert improved is False
        assert ckpt.best == first_best
        assert os.path.exists(ckpt_path)  # still only first file

def test_checkpoint_updates_best():
    model = nn.Linear(10, 1)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "best.pt")
        ckpt = ModelCheckpoint(ckpt_path, mode="min")

        ckpt.step(0.4, model)
        assert ckpt.best == 0.4

        ckpt.step(0.3, model)
        assert ckpt.best == 0.3  # updated


"""
| Component          | How to test                            | Why                                |
| ------------------ | -------------------------------------- | ---------------------------------- |
| **Optimizer**      | Compare type + hyperparameters         | Ensures config is correct          |
| **Scheduler**      | Step over epochs → LR drops            | Validates LR scheduling logic      |
| **Early stopping** | Monkeypatch losses → check epoch count | Ensures stopping triggers properly |
| **Checkpointing**  | Save → load → compare weights          | Guarantees reproducibility         |
"""