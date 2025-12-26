from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.training.registry import TrainerRegistry
from qm9_ml.utils.logging import logging

# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")


    print(cfg)

    # params for training
    cfg_training = OmegaConf.load(f"config/training/{model_type}.yaml")

    print(cfg_training)

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
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=cfg.shared.epochs,
        )

    results = trainer.train(cfg_training)

    # logging (do an appr. logging later!!!!)
    # logging(cfg.exp_name, cfg.run_tuning_name, artifacts, results, model_type, trials_data, importances)

    # if this is the best model so far, save it



    # select best model
    """
    from qm9_ml.utils.logging import select_best_model
    res = select_best_model('qm9')
    best_run_id = res['run_id']

    export_best_model(
        run_id=best_run_id,
        dst="../api-repo/models/best_model"
    )
    """


if __name__ == "__main__":
    main()


# python -m scripts.training -m model_type=mlp training=mlp
#python -m scripts.training -m model_type=mlp training=mlp training.hidden=120 shared.epochs=4