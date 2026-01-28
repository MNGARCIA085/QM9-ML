# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig
from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.training.registry import TrainerRegistry
from qm9_ml.inference.registry import PredictorRegistry
from qm9_ml.utils.logging import logging
from qm9_ml.utils.metrics import compute_metrics



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type.name
    print(f"\nSelected model: {model_type}")

    # preprocessing
    prep = PreprocessorRegistry.create(
        model_type,
        val_ratio=cfg.preprocessor.val_ratio,
        target=cfg.preprocessor.target,
        subset=cfg.preprocessor.subset,
    )
    train_ds, val_ds = prep.preprocess()

    artifacts = prep.get_artifacts() # for later logging

    # train 
    trainer = TrainerRegistry.create(
            model_type,
        )

    results = trainer.train(cfg.model_type.training, train_ds, val_ds, cfg.shared.epochs)


    # compute metrics
    predictor = PredictorRegistry.create(model_type, model=results['model'])

    y_true_train, y_pred_train = predictor.predict_with_targets(train_ds)
    y_true_val, y_pred_val = predictor.predict_with_targets(val_ds)
    
    train_metrics = compute_metrics(y_true_train, y_pred_train)
    val_metrics = compute_metrics(y_true_val, y_pred_val)

    # add metrics to results
    results['train']['metrics'] = train_metrics
    results['val']['metrics'] = val_metrics


    # logging
    logging(cfg.exp_name, cfg.run_training_name, artifacts, results, model_type)


    
    

if __name__ == "__main__":
    main()


# python -m scripts.training -m model_type=mlp training=mlp
# python -m scripts.training -m model_type=mlp training=mlp training.hidden=120 shared.epochs=4