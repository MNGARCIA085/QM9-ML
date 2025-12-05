# tests/preprocessors/test_gcn_preprocessor.py
from src.preprocessors.gcn import GCNPreprocessor
from tests.conftest import DummyDataset

def test_format_dataset_gcn(tmp_path):
    prep = GCNPreprocessor(dataset_cls=DummyDataset, root=tmp_path, last=0)
    dataset = prep._load_dataset()
    processed = prep._format_dataset(dataset, is_inference=False)

    d = processed[0]

    for attr in ["z", "pos", "edge_index", "y"]:
        assert attr in d

    # y must be sliced correctly
    assert d.y.shape == (1, 1)
