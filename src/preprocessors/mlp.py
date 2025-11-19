from .base import BasePreprocessor
from .registry import PreprocessorRegistry



@PreprocessorRegistry.register("mlp")
class MLPPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # MLP doesn't need graph transforms
        super().__init__(transform=None, **kwargs)


    """
    def _format_dataset(self, dataset, is_inference): # better namimg?????
        Extracts atomic numbers (z) and (optionally) the target (y).
        
        # We assume the base dataset returns Data objects, and we create new ones
        # with only the necessary features (z for MLP input).
        
        if is_inference:
            return [type(d)(z=d.z) for d in dataset]
        else:
            # Safely slice the target column and unsqueeze for correct shape
            target_col = self.target
            return [type(d)(z=d.z, y=d.y[:, target_col].unsqueeze(1)) for d in dataset]
    """

    def _format_dataset(self, dataset, is_inference):
        target_col = self.target
        out = []

        for d in dataset:
            d_new = d.clone()        # keeps num_nodes, graph structure, internal indexing

            # overwrite fields you want to keep
            d_new.z = d.z
            
            if not is_inference:
                d_new.y = d.y[:, target_col].unsqueeze(1)
            else:
                if hasattr(d_new, "y"):
                    del d_new.y

            out.append(d_new)

        return out

