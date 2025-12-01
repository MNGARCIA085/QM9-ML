from torch_geometric.transforms import RadiusGraph
from src.preprocessors.base import BasePreprocessor
from .registry import PreprocessorRegistry


@PreprocessorRegistry.register("gcn")
class GCNPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # Get user-supplied transform or use the default RadiusGraph
        radius = kwargs.pop("radius", 1.5)   # extracted only for GCN
        transform = RadiusGraph(r=radius)
        super().__init__(transform=transform, **kwargs)



    def _format_dataset(self, dataset, is_inference):
        """Extracts z and edge_index; and optionally the targey y"""

        target_col = self.target

        out = []

        for d in dataset:
            d_new = d.clone()

            # remove unwanted fields
            for field in ["edge_attr", "name", "smiles", "idx"]: # keep "x" and "pos" to infere num_nodes
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






"""
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
"""