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



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml")


    # preprocessing
    prep = PreprocessorRegistry.create(
        model_type,
        val_ratio=cfg.preprocessor.val_ratio,
        target=cfg.preprocessor.target,
        subset=cfg.preprocessor.subset,
    )
    train_ds, val_ds = prep.preprocess()

    artifacts = prep.get_artifacts() # for later logging

    # tuning
    tuner = TuningRegistry.create(
            model_type,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=cfg.shared.epochs,
            epochs_trials=cfg.shared.epochs_trials,
        )


    best_params, attrs = tuner.tune(n_trials=cfg.shared.num_trials,
                                    **cfg_tuning,
                                    )


    # train best model
    trainer = TrainerRegistry.create(
            model_type,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=5,
        )

    results = trainer.train_best_model(best_params)

    # logging
    logging(cfg.exp_name, cfg.run_tuning_name, artifacts, results, model_type)



    """
    model, best_params, attrs = tuner.tune(n_trials=5,
                                             batch_size_opts=[128],
                                             hidden_opts=[512],
                                             )
                                             """





if __name__ == "__main__":
    main()




"""
python -m src.scripts.tuning
python -m src.scripts.tuning model_type=tree
python -m src.scripts.tuning -m model_type=nn,tree
"""





