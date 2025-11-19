import torch
from torch.utils.data import random_split






import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, GCNConv, global_mean_pool
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch_geometric.transforms import RadiusGraph






from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]     # project/
DATA_DIR = ROOT / "data" / "QM9"





class BasePreprocessor:
    """
    Base class for data preprocessing, handling common tasks like loading, 
    subsetting, splitting, and configuration.
    """
    def __init__(self, dataset_cls=QM9, root=DATA_DIR,  #data/QM9
                 transform=None, target=0, val_ratio=0.2,
                 seed=42, subset=None):

        self.dataset_cls = dataset_cls
        self.root = root
        self.transform = transform
        self.target = target
        self.val_ratio = val_ratio
        self.seed = seed
        self.subset = subset
        # Initialize internal storage for the dataset
        self._dataset = None

    # -------------------------
    # Loading dataset (Lazy-loaded)
    # -------------------------
    def _load_dataset(self):
        """Loads the dataset once and caches it."""
        if self._dataset is None:
            # Load the dataset (expensive I/O operation)
            dataset = self.dataset_cls(root=self.root, transform=self.transform)

            if self.subset:
                # Apply subset slicing
                dataset = dataset[:self.subset]
            
            self._dataset = dataset
            
        return self._dataset

    # -------------------------
    # Split train/val
    # -------------------------
    def split(self, processed):
        """Splits the processed dataset into train and validation sets."""
        n_val = int(len(processed) * self.val_ratio)
        # Ensure n_train is the remainder to cover the whole dataset
        n_train = len(processed) - n_val 

        gen = torch.Generator().manual_seed(self.seed)
        return random_split(processed, [n_train, n_val], generator=gen)

    # -------------------------
    # Abstract/Helper
    # -------------------------
    def _get_dataset(self, dataset, is_inference):
        """
        Abstract method for model-specific data formatting and target slicing.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    # -------------------------
    # Concrete Workflow Methods (Handle Redundancy)
    # -------------------------
    
    def preprocess(self):
        """Workflow for train/validation: Load -> Format -> Split."""
        dataset = self._load_dataset()
        processed = self._get_dataset(dataset, is_inference=False)
        return self.split(processed)

    def preprocess_test(self):
        """Workflow for test: Load -> Format -> Return."""
        dataset = self._load_dataset()
        return self._get_dataset(dataset, is_inference=False)

    def preprocess_inference(self):
        """Workflow for inference: Load -> Format (no target) -> Return."""
        dataset = self._load_dataset()
        return self._get_dataset(dataset, is_inference=True)


# ---------------------------------------------------------
# MLP Preprocessor
# ---------------------------------------------------------

class MLPPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # MLP doesn't need graph transforms
        super().__init__(transform=None, **kwargs)

    def _get_dataset(self, dataset, is_inference): # better namimg?????
        """Extracts atomic numbers (z) and (optionally) the target (y)."""
        
        # We assume the base dataset returns Data objects, and we create new ones
        # with only the necessary features (z for MLP input).
        
        if is_inference:
            return [type(d)(z=d.z) for d in dataset]
        else:
            # Safely slice the target column and unsqueeze for correct shape
            target_col = self.target
            return [type(d)(z=d.z, y=d.y[:, target_col].unsqueeze(1)) for d in dataset]


# ---------------------------------------------------------
# GCN Preprocessor
# ---------------------------------------------------------

class GCNPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # Get user-supplied transform or use the default RadiusGraph
        transform = kwargs.pop("transform", RadiusGraph(r=1.5))
        super().__init__(transform=transform, **kwargs)

    """
    def _get_dataset(self, dataset, is_inference):
        Extracts atomic numbers (z), edge_index, and (optionally) the target (y).

        if is_inference:
            return [type(d)(z=d.z, edge_index=d.edge_index) for d in dataset]
        else:
            target_col = self.target
            # Requires z, edge_index, and the sliced target
            return [type(d)(z=d.z, edge_index=d.edge_index, 
                            y=d.y[:, target_col].unsqueeze(1))
                    for d in dataset]
    """
    def _get_dataset(self, dataset, is_inference):
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


# ---------------------------------------------------------
# SchNet Preprocessor
# ---------------------------------------------------------

class SchNetPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # SchNet often handles connectivity internally (e.g., using radius graph
        # and pos), so we might not need an explicit transform here, depending on 
        # how the model is implemented. Sticking with None for consistency.
        super().__init__(transform=None, **kwargs)
        
        
        
    """
    def _get_dataset(self, dataset, is_inference):

        if is_inference:
            return [type(d)(z=d.z, pos=d.pos) for d in dataset]
        else:
            target_col = self.target
            # Requires z, pos, and the sliced target
            return [type(d)(z=d.z, pos=d.pos, 
                            y=d.y[:, target_col].unsqueeze(1))
                    for d in dataset]
    """
    def _get_dataset(self, dataset, is_inference):
        target_col = self.target

        processed = []
        for d in dataset:
            d_new = d.clone()  # <-- keeps all fields: pos, z, edge_index, etc

            if is_inference:
                # remove y safely if exists
                if hasattr(d_new, "y"):
                    del d_new.y
            else:
                d_new.y = d.y[:, target_col].unsqueeze(1)

            processed.append(d_new)

        return processed