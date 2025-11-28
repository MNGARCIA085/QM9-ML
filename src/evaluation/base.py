

import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna

import mlflow
import os
import tempfile


from src.utils.metrics import compute_metrics





class BaseEvaluator(BaseTrainer):
   def __init__(self, ds, device=None):
      self.ds = ds
      self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
      super().__init__(train_ds, val_ds, test_ds, epochs=epochs, device=device) # from parent class

   def evaluate(self, loader, model):
      y_true, y_pred = self.get_predictions(loader, model)
      return compute_metrics(y_true, y_pred)


















"""
Create a BaseTrainer with:

train_one_epoch

validate

predict

Create a BaseTuner, BaseEvaluator that reuse those.

Create subclasses:

MLPTrainer, GCNTrainer, etc.

Or better: pass the model class as an argument instead of hardcoding.
"""

"""
BaseTrainer  <-- contains train(), validate(), predict()
    ↳ BaseEvaluator (inherits only predict())
    ↳ BaseTuner (inherits train/eval/predict)
"""


"""
src/
   training/
      base_trainer.py       <-- train(), validate(), predict()
      tuner.py              <-- inherits BaseTrainer
   evaluation/
      base_evaluator.py     <-- inherits BaseTrainer *or* BasePredictor
   models/
      base_model.py
      mlp.py
      gcn.py
"""