import torch
from src.preprocessors.base import BasePreprocessor
from tests.conftest import DummyDataset



class DummyPrep(BasePreprocessor):
    def _format_dataset(self, dataset, is_inference):
        # return dataset unchanged so we can test only the workflow
        return dataset


def test_load_dataset_once(tmp_path):
    prep = DummyPrep(dataset_cls=DummyDataset, root=tmp_path)
    d1 = prep._load_dataset()
    d2 = prep._load_dataset()
    assert d1 is d2, "Dataset must load only once and be cached"


def test_preprocess_returns_train_val_split(tmp_path):
    prep = DummyPrep(dataset_cls=DummyDataset, root=tmp_path)

    dataset = prep._load_dataset()
    total = len(dataset)

    train, val = prep.preprocess()

    val_expected = int(total * prep.val_ratio)
    train_expected = total - val_expected

    assert len(train) == train_expected
    assert len(val) == val_expected

    # basic sanity check
    assert hasattr(train, "__getitem__")



def test_preprocess_test_returns_list(tmp_path):
    prep = DummyPrep(dataset_cls=DummyDataset, root=tmp_path)

    dataset = prep._load_dataset()

    out = prep.preprocess_test()
    
    assert len(out) == len(dataset)
    assert isinstance(out[0], type(out[1])) # checks out[0] is the same class as out[1]


def test_preprocess_inference_flag(tmp_path, mocker):
    prep = DummyPrep(dataset_cls=DummyDataset, root=tmp_path)

    spy = mocker.spy(prep, "_format_dataset")
    prep.preprocess_inference()

    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["is_inference"] is True
