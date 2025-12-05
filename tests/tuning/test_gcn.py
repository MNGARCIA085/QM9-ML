import torch
import optuna
from src.tuning.gcn import GCNTuner
from src.tuning.registry import TuningRegistry
from torch_geometric.data import Data

# -----------------------
# Helper dataset
# -----------------------
def make_dummy_gcn_dataset(n=10):
    data_list = []
    for _ in range(n):
        num_nodes = 5

        x = torch.randn(num_nodes, 3)  # node features
        edge_index = torch.tensor([[0, 1, 2, 3],
                                   [1, 2, 3, 4]], dtype=torch.long)

        z = torch.randint(1, 10, (num_nodes,))  # fake atomic numbers / node types

        y = torch.randn(1)  # regression target

        data = Data(
            x=x,
            edge_index=edge_index,
            z=z,       # REQUIRED for your trainer
            y=y
        )

        data_list.append(data)

    return data_list


# -----------------------
# Registry test
# -----------------------
def test_gcn_tuner_is_registered():
    assert "gcn" in TuningRegistry._registry
    assert TuningRegistry.get("gcn") is GCNTuner


# -----------------------
# create_model
# -----------------------
def test_create_model_sampling():
    train_ds = make_dummy_gcn_dataset()
    val_ds = make_dummy_gcn_dataset()

    tuner = GCNTuner(train_ds, val_ds)

    trial = optuna.trial.FixedTrial({"hidden": 64})
    model = tuner.create_model(trial, hidden_opts=[32, 64, 128])

    assert model is not None
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")


# -----------------------
# objective() output
# -----------------------
def test_objective_returns_float_and_sets_metrics():
    train_ds = make_dummy_gcn_dataset()
    val_ds = make_dummy_gcn_dataset()

    tuner = GCNTuner(train_ds, val_ds, epochs_trials=1)

    trial = optuna.trial.FixedTrial({"hidden": 32, "batch_size": 16, "lr": 1e-3})

    val_loss = tuner.objective(trial)

    assert isinstance(val_loss, float)
    assert "metrics" in trial.user_attrs


# -----------------------
# Full trial run
# -----------------------
def test_gcn_tuner_full_trial():
    train_ds = make_dummy_gcn_dataset()
    val_ds = make_dummy_gcn_dataset()

    tuner = GCNTuner(train_ds, val_ds, epochs_trials=2)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda tr: tuner.objective(tr), n_trials=1)

    assert len(study.trials) == 1
    assert "metrics" in study.trials[0].user_attrs
