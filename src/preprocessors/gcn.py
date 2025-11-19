from torch_geometric.transforms import RadiusGraph
from src.preprocessors.base import BasePreprocessor
from .registry import PreprocessorRegistry


@PreprocessorRegistry.register("gcn")
class GCNPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # Get user-supplied transform or use the default RadiusGraph
        transform = kwargs.pop("transform", RadiusGraph(r=1.5))
        super().__init__(transform=transform, **kwargs)

    def _format_dataset(self, dataset, is_inference):
        """Extracts atomic numbers (z), edge_index, and (optionally) the target (y)."""
        target_col = self.target
        if is_inference:
            return [
                type(d)(z=d.z, pos=d.pos, edge_index=d.edge_index)
                for d in dataset
            ]
        else:
            return [
                type(d)(
                    z=d.z,
                    pos=d.pos,
                    edge_index=d.edge_index,
                    y=d.y[:, target_col].unsqueeze(1)
                )
                for d in dataset
            ]


"""
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
"""