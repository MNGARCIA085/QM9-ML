from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class Metrics:
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    ev: Optional[float] = None # new; only for NNs


@dataclass
class StageResults:
    losses: Optional[List[float]] = None # mse
    preds: Optional[List[Any]] = None # see later if I need it
    trues: Optional[List[Any]] = None # see later if I need it
    metrics: Metrics = field(default_factory=Metrics)


@dataclass
class Results:
    train: StageResults = field(default_factory=StageResults)
    val: StageResults = field(default_factory=StageResults)
    model: Optional[Any] = None # nn.Model as a type
    hyperparams: Optional[Dict] = None
    loss: Optional[float] = None

    def to_dict(self):
        return {
            "train": {
                "losses": self.train.losses,
                "metrics": vars(self.train.metrics),
                "loss": self.loss, # final loss
            },
            "val": {
                "losses": self.val.losses,
                "preds": self.val.preds,
                "trues": self.val.trues,
                "metrics": vars(self.val.metrics),
                "loss": self.loss,
            },
            "model": self.model,
            "hyperarams": self.hyperparams,
        }


@dataclass
class TestResults:
    preds: List[Any]
    labels: List[Any]
    metrics: Metrics = field(default_factory=Metrics)







