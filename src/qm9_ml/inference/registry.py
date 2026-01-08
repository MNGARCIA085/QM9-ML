from qm9_ml.core.registry import BaseRegistry
from qm9_ml.inference.base import BasePredictor

class PredictorRegistry(BaseRegistry):
    _registry = {}
    base_cls = BasePredictor
