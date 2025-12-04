from src.preprocessors.base import BasePreprocessor
from src.preprocessors.registry import PreprocessorRegistry



@PreprocessorRegistry.register("mlp")
class MLPPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # MLP doesn't need graph transforms
        super().__init__(transform=None, **kwargs)


    def _format_dataset(self, dataset, is_inference):
        """Extracts atomic numbers (z) and (optionally) the target (y)."""

        target_col = self.target

        out = []

        for d in dataset:
            d_new = d.clone()

            # remove unwanted fields
            for field in ["x", "edge_attr", "edge_index", "pos","name", "smiles", "idx"]:
                if hasattr(d_new, field):
                    delattr(d_new, field)

            # Explicitly set num_nodes
            d_new.num_nodes = d.z.size(0)

            # Handle inference/no-inference target
            if not is_inference:
                d_new.y = d.y[:, target_col].unsqueeze(1)
            else:
                if hasattr(d_new, "y"):
                    delattr(d_new, "y")

            out.append(d_new)

        return out



