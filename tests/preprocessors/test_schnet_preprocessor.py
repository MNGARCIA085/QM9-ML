from src.preprocessors.schnet import SchNetPreprocessor
from tests.conftest import DummyDataset

def test_format_dataset_schnet(tmp_path):
    prep = SchNetPreprocessor(dataset_cls=DummyDataset, root=tmp_path)
    dataset = prep._load_dataset()
    processed = prep._format_dataset(dataset, is_inference=False)

    d = processed[0]

    for attr in ["x", "z", "pos", "edge_attr", "edge_index", "y"]:
        assert attr in d
        assert hasattr(d, attr)

    # y must be sliced correctly from [1,19] → [1,1]
    assert d.y.shape == (1, 1)

