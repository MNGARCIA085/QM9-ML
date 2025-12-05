import torch
import pytest
from types import SimpleNamespace
from optuna.trial import FixedTrial

from src.tuning.mlp import MLPTuner
from src.tuning.registry import TuningRegistry
from src.models.mlp import SimpleMLP
from src.training.mlp import MLPTrainer
from torch_geometric.data import Data


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_dummy_mlp_dataset(n=20, num_nodes=5):
    data_list = []
    for _ in range(n):
        # --- minimal graph-like structure ---
        z = torch.randint(1, 10, (num_nodes,), dtype=torch.long)   # atomic numbers
        y = torch.randn(1)                                         # scalar regression target

        # MLP ignores edges, but PyG expects edge_index to exist
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        data = Data(
            z=z, 
            y=y,
            edge_index=edge_index
        )
        data_list.append(data)
    return data_list

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
def test_mlp_tuner_is_registered():
    assert "mlp" in TuningRegistry._registry
    assert TuningRegistry.get("mlp") is MLPTuner


# -----------------------------------------------------------------------------
# create_model sampling
# -----------------------------------------------------------------------------
def test_create_model_sampling():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds, device="cpu")

    trial = FixedTrial({"hidden": 64})
    model = tuner.create_model(trial, hidden_opts=[32, 64, 128])

    assert isinstance(model, SimpleMLP)
    print(model)
    assert model.hidden == 64  # must exist in your SimpleMLP


# -----------------------------------------------------------------------------
# Objective returns float + stores metrics
# -----------------------------------------------------------------------------
def test_objective_returns_float_and_sets_metrics():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds, epochs_trials=2, device="cpu")

    trial_params = {
        "batch_size": 16,
        "hidden": 32,
        "lr": 1e-3
    }
    trial = FixedTrial(trial_params)

    val_loss = tuner.objective(
        trial,
        batch_size_opts=[16],
        hidden_opts=[32],
        lr={"low": 1e-4, "high": 1e-2}
    )

    # return type
    assert isinstance(val_loss, float)

    # metrics stored
    metrics = trial.user_attrs.get("metrics")
    assert metrics is not None
    assert "mse" in metrics  # or whatever your trainer returns


# -----------------------------------------------------------------------------
# Full trial run (integration-like)
# -----------------------------------------------------------------------------
def test_mlp_tuner_full_trial():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds, epochs_trials=1, device="cpu")

    trial = FixedTrial({
        "batch_size": 16,
        "hidden": 64,
        "lr": 5e-4,
    })

    result = tuner.objective(
        trial,
        batch_size_opts=[16],
        hidden_opts=[64],
        lr={"low": 1e-4, "high": 1e-2}
    )

    assert isinstance(result, float)




"""
Checks

1. Registry correctness

Ensures "mlp" is registered.

Ensures the registry returns the correct class.

2. Model creation

create_model() correctly selects hidden dim from Optuna.

Checks SimpleMLP.hidden exists.

3. Objective logic

Returns a float.

Stores metrics in the trial (trial.set_user_attr("metrics", ...)).

4. Minimal integration test

Runs a full tuning lifecycle: loaders → trainer → run_epoch → evaluate.
"""