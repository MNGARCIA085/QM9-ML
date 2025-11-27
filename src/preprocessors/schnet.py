from .registry import PreprocessorRegistry
from src.preprocessors.base import BasePreprocessor

@PreprocessorRegistry.register("schnet")
class SchNetPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # SchNet often handles connectivity internally (e.g., using radius graph
        # and pos), so we might not need an explicit transform here, depending on 
        # how the model is implemented. Sticking with None for consistency.
        cutoff = kwargs.pop("cutoff", 10.0)
        super().__init__(transform=None, **kwargs)
        
    
    def _format_dataset(self, dataset, is_inference):
        """Extracts atomic numbers (z) , pos, and (optionally) the target (y)."""

        target_col = self.target

        out = []

        for d in dataset:
            d_new = d.clone()

            # remove unwanted fields
            for field in ["x", "edge_attr", "edge_index", "name", "smiles", "idx"]:
                if hasattr(d_new, field):
                    delattr(d_new, field)

            # Handle inference/no-inference target
            if not is_inference:
                d_new.y = d.y[:, target_col].unsqueeze(1)
            else:
                if hasattr(d_new, "y"):
                    delattr(d_new, "y")

            out.append(d_new)

        return out