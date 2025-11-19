


class PreprocessorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(prep_cls):
            cls._registry[name] = prep_cls
            return prep_cls
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown preprocessor: {name}")
        return cls._registry[name](**kwargs)


"""
prep = PreprocessorRegistry.create(model_type="gcn", target=1)
train, val = prep.preprocess()


prep = PreprocessorRegistry.create(
    model_type,
    target=0,
    root="data/QM9"
)

train, val = prep.preprocess()

python train.py model_type=gcn target=2




✔ Centralized mapping

You don’t spread “if model == X” across files.

✔ Extensible

Add a new model → implement class → add a decorator.
No other changes.

✔ Works with Hydra / YAML

Very easy to integrate into your config system.

✔ Clean separation

Preprocessors encapsulate their formatting logic

Factory/registry encapsulates the selection logic

Trainer remains simple

✔ No circular imports

Because registration happens when modules are imported


"""



"""
for later
@PreprocessorRegistry.register("gcn")
class GCNPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        radius = kwargs.pop("radius", 3.0)   # extracted only for GCN
        transform = RadiusGraph(r=radius)
        
        super().__init__(transform=transform, **kwargs)
MLP
python
Copiar código
@PreprocessorRegistry.register("mlp")
class MLPPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(transform=None, **kwargs)
SchNet
python
Copiar código
@PreprocessorRegistry.register("schnet")
class SchNetPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        cutoff = kwargs.pop("cutoff", 5.0)
        # maybe you don't even use it yet
        super().__init__(transform=None, **kwargs)
"""