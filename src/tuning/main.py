




from src.preprocessors.mlp import MLPPreprocessor
from .mlp import MLPTuner


def main():

    # prep
    prep = MLPPreprocessor(subset=50)
    train_ds, val_ds = prep.preprocess()

    # tuning
    tuner = MLPTuner(train_ds, val_ds)

    #
    best_config = tuner.tune(num_samples=2)
    print(best_config)








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





