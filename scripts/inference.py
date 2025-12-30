from torch_geometric.data import Data
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from qm9_ml.preprocessors.registry import PreprocessorRegistry
from qm9_ml.tuning.registry import TuningRegistry
from qm9_ml.training.registry import TrainerRegistry
from qm9_ml.utils.logging import logging
from qm9_ml.utils.logging import select_best_model





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    #
    exp_name = cfg.exp_name
    run_tuning_name = cfg.run_tuning_name

    # get best model (overall)
    results = select_best_model(exp_name, run_tuning_name)
    model_type = results['model_type']
    model = mlflow.pytorch.load_model(results["model_uri"])

    # trainer
    trainer = TrainerRegistry.create(
            model_type,
        )

    # ------------- Predictions-----------------
    

    print("----------One sample---------------")
    sample = Data(
                x=torch.randn(5, 11),
                edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]]),
                edge_attr=torch.randn(4, 4),
                y=torch.randn(1, 19),
                pos=torch.randn(5, 3),
                z=torch.randint(1, 10, (5,)),
                smiles="H2O",
                name="mol1",
                idx=torch.tensor([0])
            )
    aux = trainer.predict(sample, model)
    print(aux)


    print("--------Test set predictions -------")
    # load and preprocess data
    prep = PreprocessorRegistry.create(
        model_type,
    )
    test_ds = prep.preprocess_test()

    preds = trainer.predict(test_ds, model)

    print(preds.shape)
    print(preds)


    
    
if __name__ == "__main__":
    main()

