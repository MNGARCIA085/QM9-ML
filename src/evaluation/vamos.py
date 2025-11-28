import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from pathlib import Path





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







ROOT = Path(__file__).resolve().parents[2]     # project/; later it will be 1 level above (when i move this to scripts)
DATA_DIR = ROOT / "data" / "QM9"



def main():
    results = select_best_model('test', 'tuning')
    print(results["model_uri"])

    # load model (to evaluate later)
    model = mlflow.pytorch.load_model(results["model_uri"])
    print(model)

    model_type = 'schnet' # later i get this from my best model

    # fake data to make a pred....
    prep = PreprocessorRegistry.create(
        model_type,
        root=DATA_DIR,
    )


    test_ds = prep.preprocess_test()

    from torch_geometric.loader import DataLoader
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # evaluate
    for s in loader:
    	print(s)
    	break



if __name__ == "__main__":
    main()
