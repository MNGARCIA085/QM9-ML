from .mlp import MLPTuner
from .gcn import GCNTuner
from .schnet import SchNetTuner

from src.preprocessors.mlp import MLPPreprocessor
from src.preprocessors.gcn import GCNPreprocessor
from src.preprocessors.schnet import SchNetPreprocessor


def main():
    
    #prep = MLPPreprocessor(subset=1000)
    #prep = GCNPreprocessor(subset=1000)

    prep = SchNetPreprocessor(subset=1000)

    train_ds, val_ds = prep.preprocess()
    
    #tuner = MLPTuner(train_ds, val_ds)
    #tuner = GCNTuner(train_ds, val_ds)
    tuner = SchNetTuner(train_ds, val_ds)
    
    model, best_params, attrs = tuner.tune(n_trials=5)

    print(best_params) # bs, lr, hidden


    



if __name__ == "__main__":
    main()








"""
python -m src.scripts.tuning
python -m src.scripts.tuning model_type=tree
python -m src.scripts.tuning -m model_type=nn,tree
"""


# see later // training
# python train.py -m models=tree,nn,rf
# python train.py -m models=tree,nn,rf hydra/launcher=submitit_local
# python train.py -m models=tree,nn lr=0.001,0.01 batch_size=32,64





