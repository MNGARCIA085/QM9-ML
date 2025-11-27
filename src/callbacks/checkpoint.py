



# to use mlflow

import mlflow



import torch
from src.utils.logging import mlflow_save_model_artifact


class MLflowModelCheckpoint:
    """
    Callback that saves the best model to MLflow artifacts.
    """
    def __init__(self, monitor="val_loss", mode="min"):
        self.monitor = monitor
        self.mode = mode

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.best = float("inf") if mode == "min" else -float("inf")

    def step(self, metric_value, model):
        improved = (
            metric_value < self.best if self.mode == "min"
            else metric_value > self.best
        )

        if improved:
            self.best = metric_value

            # Delegate saving to MLflow utils (centralized behavior)
            mlflow_save_model_artifact(model)

            print(f"[MLflow Checkpoint] New best model "
                  f"({self.monitor}={metric_value:.4f})")









class ModelCheckpoint:
    def __init__(self, filepath="best_model.pt", monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def step(self, metric_value, model):
        better = (
            metric_value < self.best if self.mode == "min"
            else metric_value > self.best
        )
        
        if better:
            self.best = metric_value
            torch.save(model.state_dict(), self.filepath)
