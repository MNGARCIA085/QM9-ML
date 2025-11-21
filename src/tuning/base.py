import ray
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.schedulers import ASHAScheduler




class BaseTuner(ABC):
    """
    Abstract base class for Ray Tune-based hyperparameter tuning.
    Handles Ray object storage, generic tune(), and metric averaging.
    """

    def __init__(self, train_ds, val_ds):
        self.train_ds = ray.put(train_ds)
        self.val_ds = ray.put(val_ds)



    # ---------------- Abstract methods ---------------- #
    @abstractmethod
    def _train_model_ray(self, config):
        """Train one model configuration for Ray Tune."""
        pass

    @abstractmethod
    def get_tune_config(self):
        """Return the hyperparameter search space for Ray Tune."""
        pass

    """
    @abstractmethod
    def train_best_model(self, config) -> Results:
        #Train a model with the given config and return metrics + model.
        pass
    """

    # ---------------- Shared method ---------------- #
    def tune(self, num_samples=2): 
        """
        Generic Ray Tune wrapper.
        Subclasses provide _train_model_ray and get_tune_config.
        """
        config = self.get_tune_config()
        scheduler = ASHAScheduler(metric="val_loss", mode="min")

        tuner = tune.Tuner(
            tune.with_parameters(self._train_model_ray),
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=num_samples
            )
        )
        results = tuner.fit()
        best = results.get_best_result(metric="val_loss", mode="min")
        return best.config






