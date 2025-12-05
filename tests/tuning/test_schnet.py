import pytest
import torch
import optuna

from src.tuning.schnet import SchNetTuner
from src.tuning.registry import TuningRegistry
from torch_geometric.data import Data


# ---------------------------------------------------------
# Helpers: tiny synthetic dataset
# ---------------------------------------------------------
def tiny_dataset(n=10):
    data_list = []
    for _ in range(n):
        x = torch.randn(5, 4)      # node features
        pos = torch.randn(5, 3)    # positions
        z = torch.randint(1, 5, (5,))  # atomic numbers
        y = torch.randn(1)         # target

        data_list.append(Data(x=x, pos=pos, z=z, y=y, batch=torch.zeros(5, dtype=torch.long)))
    return data_list


# ---------------------------------------------------------
# 1. Registry test
# ---------------------------------------------------------
def test_schnet_tuner_is_registered():
    assert "schnet" in TuningRegistry._registry
    assert TuningRegistry.get("schnet") is SchNetTuner



# ---------------------------------------------------------
# 2. Test create_model sampling
# ---------------------------------------------------------
def test_create_model_sampling():
    train_ds = tiny_dataset()
    val_ds = tiny_dataset()

    tuner = SchNetTuner(train_ds, val_ds, device="cpu")

    def _objective(trial):
        model = tuner.create_model(
            trial,
            hidden_channels_opts=[32, 64],
            num_filters_opts=[32, 64],
            num_interactions_low=3,
            num_interactions_high=5
        )
        print(model)
        assert model.schnet.hidden_channels in [32, 64]
        assert model.schnet.num_filters in [32, 64]
        assert 3 <= model.schnet.num_interactions <= 5
        return 1.0  # dummy

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=1)


# ---------------------------------------------------------
# 3. Test objective returns float and sets metrics
# ---------------------------------------------------------
def test_objective_returns_float_and_sets_metrics():
    train_ds = tiny_dataset()
    val_ds = tiny_dataset()

    tuner = SchNetTuner(train_ds, val_ds, epochs_trials=1, device="cpu")

    def _objective_wrapper(trial):
        loss = tuner.objective(
            trial,
            hidden_channels_opts=[8, 16],
            num_filters_opts=[8, 16],
            num_interactions={"low": 1, "high": 2},
            batch_size_opts=[4],
            lr={"low": 1e-4, "high": 1e-3},
        )
        # Check the return type
        assert isinstance(loss, float)
        # Check that metrics are stored
        assert "metrics" in trial.user_attrs
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective_wrapper, n_trials=1)


# ---------------------------------------------------------
# 4. Integration: run a full single-trial tuning
# ---------------------------------------------------------
def test_schnet_tuner_full_trial():
    train_ds = tiny_dataset()
    val_ds = tiny_dataset()

    tuner = SchNetTuner(train_ds, val_ds, epochs_trials=1, device="cpu")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: tuner.objective(trial), n_trials=1)

    assert len(study.trials) == 1
    trial = study.trials[0]

    assert isinstance(trial.value, float)
    assert "metrics" in trial.user_attrs



"""
def test_schnet_tuner_runs(prep_small):
    train_ds, val_ds = prep_small.preprocess()

    tuner = SchNetTuner(train_ds, val_ds, epochs_trials=1, device="cpu")

    # Small search space so it runs fast
    params, attrs, trials = tuner.tune(
        n_trials=1,
        hidden_channels_opts=[16],
        num_filters_opts=[16],
        num_interactions={"low": 2, "high": 2},
        batch_size_opts=[8],
        lr={"low": 1e-4, "high": 1e-3}
    )

    assert isinstance(params, dict)
    assert isinstance(attrs, dict)
    assert isinstance(trials, list)
    assert "lr" in params
    assert len(trials) == 1
"""