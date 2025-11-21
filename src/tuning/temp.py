# Preprocess once (large subset)
prep = PreprocessorRegistry.create("mlp", root=DATA_DIR, subset=5000)
dataset = prep.preprocess()

# Put in Ray object store
dataset_ref = ray.put(dataset)


def train_ray(config):
    prep = PreprocessorRegistry.create("mlp", root=DATA_DIR, subset=config["subset"])
    train_loader, val_loader = loaders(batch_size=config["batch_size"], subset=config["subset"])
