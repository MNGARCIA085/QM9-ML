import pytest
from src.preprocessors.registry import PreprocessorRegistry

# I need this to test registry
from src.preprocessors.mlp import MLPPreprocessor
from src.preprocessors.gcn import GCNPreprocessor
from src.preprocessors.schnet import SchNetPreprocessor


def test_registry_resolves_classes():
    names = PreprocessorRegistry._registry.keys()
    assert "mlp" in names
    assert "gcn" in names
    assert "schnet" in names

def test_registry_create():
    pre = PreprocessorRegistry.create("mlp", target=0)
    from src.preprocessors.mlp import MLPPreprocessor
    assert isinstance(pre, MLPPreprocessor)

def test_registry_unknown():
    with pytest.raises(ValueError):
        PreprocessorRegistry.create("does_not_exist")
