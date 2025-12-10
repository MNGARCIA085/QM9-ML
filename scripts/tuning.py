from src.preprocessors.registry import PreprocessorRegistry
from src.tuning.registry import TuningRegistry
from src.training.registry import TrainerRegistry

from src.utils.logging import logging

# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")

    cfg_tuning = OmegaConf.load(f"config/tuning/{model_type}.yaml")

    #print("Registry:", PreprocessorRegistry._registry)

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


    best_params, attrs, trials_data = tuner.tune(n_trials=cfg.shared.num_trials,
                                    **cfg_tuning,
                                    )


    # train best model
    trainer = TrainerRegistry.create(
            model_type,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=cfg.shared.epochs,
        )

    results = trainer.train(best_params)

    # logging
    logging(cfg.exp_name, cfg.run_tuning_name, artifacts, results, model_type, trials_data)




    # improve later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






    """

    # save best model to the appropiate folder (for the API); simulate a registry
    from src.utils.logging import select_best_model
    from pathlib import Path
    import mlflow
    import torch

    exp_name = cfg.exp_name
    run_tuning_name = cfg.run_tuning_name

    root_dir = Path(__file__).resolve().parents[1]
    PATH = root_dir / "api" / "model"

    # Ensure directory exists
    PATH.mkdir(parents=True, exist_ok=True)

    results = select_best_model(exp_name, run_tuning_name)

    model = mlflow.pytorch.load_model(results["model_uri"])
    model.eval()

    save_path = PATH / "model.pt"
    torch.save(model.state_dict(), save_path) # only weights

    print("Model saved to:", save_path)

    """


    # save_model_for_api




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





"""
# create an uninitialized model with the SAME architecture
model = MLPRegressor(**same_params_as_training)

# load weights
state_dict = torch.load("api/model/model.pt", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

"""