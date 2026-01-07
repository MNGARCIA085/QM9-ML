import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qm9_ml.training.schnet import SchNetTrainer
from qm9_ml.inference.schnet import SchNetPredictor


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

    trainer = SchNetTrainer()
    model = trainer.create_model_from_params(params)
    predictor = SchNetPredictor(model=model)
    
    y_true, y_pred = predictor.predict_with_targets(loader)

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

    trainer = SchNetTrainer()
    model = trainer.create_model_from_params(params)
    predictor = SchNetPredictor(model=model)
    
    preds = predictor.predict(loader)

    assert preds.dim() == 1
    assert len(preds) == len(ds)
