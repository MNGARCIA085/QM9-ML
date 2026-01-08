from qm9_ml.core.registry import BaseRegistry
from qm9_ml.preprocessors.base import BasePreprocessor

class PreprocessorRegistry(BaseRegistry):
    _registry = {}
    base_cls = BasePreprocessor

