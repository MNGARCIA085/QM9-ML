import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from torch_geometric.data import Batch


#from ray.data.datasource import PythonSplitter

import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from torch_geometric.data import Batch


# ---------------------------------------------------------
# Wrapper so Ray cannot infer Arrow schema
# ---------------------------------------------------------
class GraphWrapper:
    def __init__(self, g):
        self.g = g


# ---------------------------------------------------------
# Base tuner
# ---------------------------------------------------------
import torch
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler


class BaseTuner(ABC):
    """
    Clean Ray Tune hyperparameter tuner for PyG datasets.
    Uses plain Python lists and PyG DataLoaders per trial.
    """

    def __init__(self, train_ds, val_ds):
        # Just store the raw PyG Data objects
        self.train_list = train_ds
        self.val_list = val_ds

    def create_loaders(self, batch_size):
        train_loader = DataLoader(
            self.train_list,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            self.val_list,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    # ---------------- Abstract methods ---------------- #
    @abstractmethod
    def _train_model_ray(self, config):
        """Train one model config."""
        pass

    @abstractmethod
    def get_tune_config(self):
        """Return Ray Tune search space."""
        pass

    # -----------------------------------------------------
    # Ray Tune wrapper
    # -----------------------------------------------------
    def tune(self, num_samples=2):
        config = self.get_tune_config()
        scheduler = ASHAScheduler(metric="val_loss", mode="min")

        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=num_samples,
            )
        )
        results = tuner.fit()
        best = results.get_best_result(metric="val_loss", mode="min")
        return best.config
