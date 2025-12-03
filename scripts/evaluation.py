import hydra
from omegaconf import DictConfig, OmegaConf
from src.preprocessors.registry import PreprocessorRegistry
from src.preprocessors.mlp import MLPPreprocessor
from src.preprocessors.gcn import GCNPreprocessor
from src.preprocessors.schnet import SchNetPreprocessor
from src.tuning.registry import TuningRegistry
from src.tuning.mlp import MLPTuner
from src.tuning.gcn import GCNTuner
from src.tuning.schnet import SchNetTuner
from src.utils.logging import logging


from src.training.registry import TrainerRegistry
from src.training.mlp import MLPTrainer
from src.training.gcn import GCNTrainer
from src.training.schnet import SchNetTrainer




from src.utils.logging import select_best_model, log_test_results

import mlflow


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

    # --- evaluate (using the appropiate trainer) ---

    # trainer
    trainer = TrainerRegistry.create(
            model_type,
        )

    # evaluate
    metrics = trainer.evaluate(test_ds, model)

    print(len(test_ds))

    print(metrics)

    # log best model results
    log_test_results(exp_name, results["run_id"], model_type, metrics)


    
    
if __name__ == "__main__":
    main()




"""
python -m src.scripts.tuning
python -m src.scripts.tuning model_type=tree
python -m src.scripts.tuning -m model_type=nn,tree
"""





