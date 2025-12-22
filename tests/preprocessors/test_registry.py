import pytest
from qm9_ml.preprocessors.registry import PreprocessorRegistry

# I need this to test registry
from qm9_ml.preprocessors.mlp import MLPPreprocessor
from qm9_ml.preprocessors.gcn import GCNPreprocessor
from qm9_ml.preprocessors.schnet import SchNetPreprocessor


def test_registry_resolves_classes():
    names = PreprocessorRegistry._registry.keys()
    assert "mlp" in names
    assert "gcn" in names
    assert "schnet" in names

def test_registry_create():
    pre = PreprocessorRegistry.create("mlp", target=0)
    from qm9_ml.preprocessors.mlp import MLPPreprocessor
    assert isinstance(pre, MLPPreprocessor)

def test_registry_unknown():
    with pytest.raises(ValueError):
        PreprocessorRegistry.create("does_not_exist")
