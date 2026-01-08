import torch
from torch_geometric.loader import DataLoader
from qm9_ml.training.schnet import SchNetTrainer
from qm9_ml.inference.schnet import SchNetPredictor



# ------------------------------------------------------------
# Test: get_predictions
# ------------------------------------------------------------
def test_schnet_predictions_with_targets(dataset_schnet):
    ds = dataset_schnet(12, 5)
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
def test_schnet_predict(dataset_schnet):
    ds = dataset_schnet(12, 5)
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
