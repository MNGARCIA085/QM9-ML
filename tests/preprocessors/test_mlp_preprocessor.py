import torch
from src.preprocessors.mlp import MLPPreprocessor
from tests.conftest import DummyDataset

def test_format_dataset_mlp(tmp_path):
    prep = MLPPreprocessor(dataset_cls=DummyDataset, root=tmp_path, last=0)

    dataset = prep._load_dataset()
    processed = prep._format_dataset(dataset, is_inference=False)

    d = processed[0]


    assert "z" in d
    assert "y" in d
    assert "edge_index" not in d
    assert d.y.shape == (1, 1) # original y.shape == (1, 19), but we pick only 1 target
    assert d.z.shape == (d.num_nodes,) # z is atomic number per atom in the molecule


def test_mlp_inference_removes_targets(tmp_path):
    prep = MLPPreprocessor(dataset_cls=DummyDataset, root=tmp_path, last=0)
    
    dataset = prep._load_dataset()
    processed = prep._format_dataset(dataset, is_inference=True)
    
    d = processed[0]
    
    #print(d.keys())
    assert "y" not in d, "Inference should not have targets"


