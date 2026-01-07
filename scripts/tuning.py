
# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf
from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.tuning.registry import TuningRegistry
from qm9_ml.training.registry import TrainerRegistry
from qm9_ml.inference.registry import PredictorRegistry
from qm9_ml.utils.logging import logging, select_best_model, export_best_model
from qm9_ml.utils.metrics import compute_metrics



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml")

    #print("Registry:", PreprocessorRegistry._registry), registry check

    # preprocessing
    prep = PreprocessorRegistry.create(
        model_type,
        val_ratio=cfg.preprocessor.val_ratio,
        target=cfg.preprocessor.target,
        subset=cfg.preprocessor.subset,
    )
    train_ds, val_ds = prep.preprocess()

    print(type(train_ds))

    for x in train_ds:
        print(x)
        break


    artifacts = prep.get_artifacts() # for later logging

    # tuning
    tuner = TuningRegistry.create(
            model_type,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=cfg.shared.epochs,
            epochs_trials=cfg.shared.epochs_trials,
        )


    best_params, attrs, trials_data, importances = tuner.tune(n_trials=cfg.shared.num_trials,
                                    **cfg_tuning,
                                    )


    # train best model
    trainer = TrainerRegistry.create(
            model_type,
        )

    results = trainer.train(best_params, train_ds, val_ds, cfg.shared.epochs)

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
    logging(cfg.exp_name, cfg.run_tuning_name, artifacts, results, model_type, trials_data, importances)


    # select and then save best model (simplistic way: always expport the best)
    res = select_best_model('qm9')
    best_run_id = res['run_id'] # maybe add later model_type
    export_best_model(
        run_id=best_run_id,
        dst="../api-repo/models/best_model"
    )



if __name__ == "__main__":
    main()

