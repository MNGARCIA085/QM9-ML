


from .mlp_tuner import MLPTuner
# from gcn_tuner import GCNTuner
# from schnet_tuner import SchNetTuner

from src.preprocessors.mlp import MLPPreprocessor


def main():
    prep = MLPPreprocessor(subset=1000)
    train_ds, val_ds = prep.preprocess()
    tuner = MLPTuner(train_ds, val_ds)
    best_params = tuner.tune(n_trials=8)


if __name__ == "__main__":
    main()

