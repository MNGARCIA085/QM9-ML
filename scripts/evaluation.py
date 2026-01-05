import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.inference.registry import PredictorRegistry
from qm9_ml.utils.logging import logging,select_best_model, log_test_results
from qm9_ml.utils.metrics import compute_metrics




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    #
    exp_name = cfg.exp_name
    run_tuning_name = cfg.run_tuning_name


    # get best model (overall)
    results = select_best_model(exp_name, run_tuning_name)
    model_type = results['model_type']
    model = mlflow.pytorch.load_model(results["model_uri"])

    # load and preprocess test data
    prep = PreprocessorRegistry.create(
        model_type,
    )
    test_ds = prep.preprocess_test()

    # --- evaluate (using the appropiate predictor) ---

    # predictor
    predictor = PredictorRegistry.create(
            model_type,
            model=model,
        )
    y_true, y_pred = predictor.predict_with_targets(test_ds)

    # metrics
    metrics = compute_metrics(y_true, y_pred)


    print(len(test_ds))

    print(metrics)


    # log best model results
    log_test_results(exp_name, results["run_id"], model_type, metrics, model.config)


    
    
if __name__ == "__main__":
    main()

