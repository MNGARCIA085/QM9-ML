from qm9_ml.core.registry import BaseRegistry
from qm9_ml.tuning.base import BaseTuner

class TuningRegistry(BaseRegistry):
    _registry = {}
    base_cls = BaseTuner
