import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from qm9_ml.inference.mlp import MLPPredictor



def test_predictiosns_with_targets(dataset_mlp, mock_mlp):
    loader = DataLoader(dataset_mlp(4), batch_size=1, shuffle=False)
    model = mock_mlp
    predictor = MLPPredictor(model=model, device="cpu")

    trues, preds = predictor.predict_with_targets(loader)

    assert trues.shape == preds.shape == torch.Size([4])
    assert trues.ndim == preds.ndim == 1



def test_predict(dataset_mlp, mock_mlp):
    loader = DataLoader(dataset_mlp(3), batch_size=1, shuffle=False)
    model = mock_mlp
    predictor = MLPPredictor(model=model, device="cpu")

    preds = predictor.predict(loader)

    assert preds.shape == torch.Size([3])
    assert preds.ndim == 1


