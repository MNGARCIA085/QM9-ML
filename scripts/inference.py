from src.preprocessors.registry import PreprocessorRegistry
from src.tuning.registry import TuningRegistry
from src.training.registry import TrainerRegistry

from src.utils.logging import logging
from src.utils.logging import select_best_model

import mlflow

import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    #
    exp_name = cfg.exp_name
    run_tuning_name = cfg.run_tuning_name

    # get best model (overall)
    results = select_best_model(exp_name, run_tuning_name)
    model_type = results['model_type']
    model = mlflow.pytorch.load_model(results["model_uri"])

    # load and preprocess inference data; to simplify we will use test set
    prep = PreprocessorRegistry.create(
        model_type,
    )
    test_ds = prep.preprocess_test()

    # --- evaluate (using the appropiate trainer) ---

    # trainer
    trainer = TrainerRegistry.create(
            model_type,
        )

    # predictions
    preds = trainer.predict(test_ds, model)

    print(preds.shape)
    print(preds)


    
    
if __name__ == "__main__":
    main()

