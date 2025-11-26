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

# check if i can load best model
from src.utils.logging import select_best_model

import mlflow


def main():
    results = select_best_model('test', 'tuning')
    print(results["model_uri"])

    # load model (to evaluate later)
    model = mlflow.pytorch.load_model(results["model_uri"])
    print(model)

    model_type = 'schnet'

    # fake data to make a pred....
    prep = PreprocessorRegistry.create(
        model_type,
        target=0,
        subset=1000,
    )
    train_ds, val_ds = prep.preprocess()

    tuner = TuningRegistry.create(
        model_type,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_ds, batch_size=32, shuffle=False)

    trues, preds = tuner.get_predictions(loader, model)

    print(trues.shape, preds.shape)


if __name__ == "__main__":
    main()
