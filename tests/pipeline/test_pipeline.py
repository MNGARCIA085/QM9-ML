import pytest
from omegaconf import OmegaConf

from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.training.registry import TrainerRegistry
from qm9_ml.tuning.registry import TuningRegistry
from qm9_ml.inference.registry import PredictorRegistry


from qm9_ml.utils.metrics import compute_metrics


@pytest.fixture
def tiny_cfg():
    return OmegaConf.create(
        {
            "model_type": "schnet",

            "preprocessor": {
                "val_ratio": 0.2,
                "target": 0,
                "subset": 50,
            },

            "shared": {
                "epochs": 1,
                "epochs_trials": 1,
                "num_trials": 2,
            },

            "exp_name": "test-exp",
            "run_tuning_name": "test-run",
        }
    )


@pytest.fixture
def tiny_tuning_cfg():
    return OmegaConf.create(
        {
            "hidden_channels_opts": [16],
            "num_filters_opts": [16],
            "num_interactions": {"low": 2, "high": 2},
            "batch_size_opts": [8],
            "lr": {"low": 1e-4, "high": 1e-3},
        }
    )


def test_full_pipeline_integration(tiny_cfg, tiny_tuning_cfg):
    """
    Full end-to-end integration test:
    preprocess → tune → train best model → evaluate final model.
    """

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------
    prep = PreprocessorRegistry.create(
        tiny_cfg.model_type,
        val_ratio=tiny_cfg.preprocessor.val_ratio,
        target=tiny_cfg.preprocessor.target,
        subset=tiny_cfg.preprocessor.subset,
    )

    train_ds, val_ds = prep.preprocess()

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    # Create a small test split from val_ds (fast)
    from torch.utils.data import Subset
    test_ds = Subset(val_ds.dataset, val_ds.indices[:10])

    #test_ds = val_ds[:10]

    artifacts = prep.get_artifacts()

    # --------------------------------------------------
    # TUNING
    # --------------------------------------------------
    tuner = TuningRegistry.create(
        tiny_cfg.model_type,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=tiny_cfg.shared.epochs,
        epochs_trials=tiny_cfg.shared.epochs_trials,
    )

    best_params, attrs, trials_data, importances = tuner.tune(
        n_trials=tiny_cfg.shared.num_trials,
        **tiny_tuning_cfg
    )

    assert isinstance(best_params, dict)
    assert isinstance(trials_data, list)

    # --------------------------------------------------
    # TRAIN FINAL MODEL
    # --------------------------------------------------
    trainer = TrainerRegistry.create(
        tiny_cfg.model_type,
        
    )

    results = trainer.train(
        best_params,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=tiny_cfg.shared.epochs,
    )

    assert isinstance(results, dict)
    assert len(results) > 0

    # Get trained model
    model = results.get("model", None)
    assert model is not None, "Trainer should return the trained model."



    
    # --------------------------------------------------
    # EVALUATE FINAL MODEL 
    # --------------------------------------------------
    predictor = PredictorRegistry.create(
            tiny_cfg.model_type,
            model=model,
        )
    y_true, y_pred = predictor.predict_with_targets(test_ds)

    # metrics
    metrics = compute_metrics(y_true, y_pred)



    #metrics = trainer.evaluate(test_ds, model)

    assert isinstance(metrics, dict), "evaluate() should return a dict of metrics."
    assert len(metrics) > 0, "Evaluation metrics dict should not be empty."
    



