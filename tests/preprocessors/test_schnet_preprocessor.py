from src.preprocessors.schnet import SchNetPreprocessor
from tests.conftest import DummyDataset

def test_format_dataset_schnet(tmp_path):
    prep = SchNetPreprocessor(dataset_cls=DummyDataset, root=tmp_path, last=0)
    dataset = prep._load_dataset()
    processed = prep._format_dataset(dataset, is_inference=False)

    d = processed[0]

    # later analyze full list and use whgite flag in prep for feat. ["x", "z", "pos", "edge_attr", "edge_index", "y"]
    for attr in ["z", "pos", "y"]:
        assert attr in d
        assert hasattr(d, attr)

    # y must be sliced correctly from [1,19] → [1,1]
    assert d.y.shape == (1, 1)

