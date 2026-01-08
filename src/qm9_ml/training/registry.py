from qm9_ml.core.registry import BaseRegistry
from qm9_ml.training.base import BaseTrainer # not sctrictly needed but gives structural guarantees

class TrainerRegistry(BaseRegistry):
    _registry = {}
    base_cls = BaseTrainer

