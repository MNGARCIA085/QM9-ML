import torch
import optuna
from src.tuning.mlp import MLPTuner
from src.tuning.registry import TuningRegistry
from torch_geometric.data import Data

# -----------------------
# Helper dataset
# -----------------------
def make_dummy_mlp_dataset(n=10):
    data_list = []
    for _ in range(n):
        num_nodes = 5

        z = torch.randint(0, 10, (num_nodes,))       # categorical node types
        batch = torch.zeros(num_nodes, dtype=torch.long)

        data = Data(
            z=z,
            batch=batch,
            y=torch.randn(1)
        )
        data_list.append(data)

    return data_list


# -----------------------
# Registry test
# -----------------------
def test_mlp_tuner_is_registered():
    assert "mlp" in TuningRegistry._registry
    assert TuningRegistry.get("mlp") is MLPTuner


# -----------------------
# create_model
# -----------------------
def test_create_model_sampling():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds)

    trial = optuna.trial.FixedTrial({"hidden": 64})
    model = tuner.create_model(trial, hidden_opts=[32, 64, 128])

    assert model is not None
    # generic check for Linear layers
    assert any(isinstance(m, torch.nn.Linear) for m in model.modules())


# -----------------------
# objective() output
# -----------------------
def test_objective_returns_float_and_sets_metrics():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds, epochs_trials=1)

    trial = optuna.trial.FixedTrial({
        "hidden": 32,
        "batch_size": 16,
        "lr": 1e-3
    })

    val_loss = tuner.objective(trial)

    assert isinstance(val_loss, float)
    assert "metrics" in trial.user_attrs


# -----------------------
# Full trial run
# -----------------------
def test_mlp_tuner_full_trial():
    train_ds = make_dummy_mlp_dataset()
    val_ds = make_dummy_mlp_dataset()

    tuner = MLPTuner(train_ds, val_ds, epochs_trials=2)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda tr: tuner.objective(tr), n_trials=1)

    assert len(study.trials) == 1
    assert "metrics" in study.trials[0].user_attrs
