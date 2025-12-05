import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]     # project/
DATA_DIR = ROOT / "data" / "QM9"


#https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html





class BasePreprocessor:
    """
    Base class for data preprocessing, handling common tasks like loading, 
    subsetting, splitting, and configuration.

    This class provides a simple and reproducible way to work with the QM9 dataset
    while keeping a fixed test split for final evaluation.

    The dataset is loaded once for training and hyperparameter tuning, where all
    samples except the last N are used. The final "last" (200 by default) molecules of the dataset
    are reserved as a held-out test set to simulate a real-world evaluation
    scenario. During training/tuning only the training subset is loaded; the test
    subset is loaded separately and only when needed for final evaluation.

    """
    def __init__(self, dataset_cls=QM9, root=DATA_DIR,
                 transform=None, target=0, val_ratio=0.2,
                 seed=42, last=200, subset=None):

        self.dataset_cls = dataset_cls
        self.root = root
        self.transform = transform
        self.target = target
        self.val_ratio = val_ratio
        self.seed = seed
        self.subset = subset
        # Initialize internal storage for the dataset
        self._dataset = None
        self._test_dataset = None
        # to separate for test
        self.last = last

    # -------------------------
    # Loading dataset (Lazy-loaded)
    # -------------------------


    def _load_dataset(self):
        if self._dataset is None:
            dataset = self.dataset_cls(root=self.root, transform=self.transform)
            full = dataset

            if self.subset:
                train_part = full[:-self.last] if self.last > 0 else full
                dataset = train_part[:self.subset]
            else:
                dataset = full if self.last == 0 else full[:-self.last]

            self._dataset = dataset

        return self._dataset


    # alwasy load tiny version
    def _load_test_dataset(self):
        """Load only the test slice (last samples)."""
        if self._test_dataset is None:
            dataset = self.dataset_cls(root=self.root, transform=self.transform)
            self._test_dataset = dataset[-self.last:]

        return self._test_dataset

    
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
    def _format_dataset(self, dataset, is_inference):
        """
        Abstract method for model-specific data formatting and target slicing.
        Must be implemented by subclasses.
          For MLP: extract z, and a column of y
          For GCN: extract z, pos, edge_index, y
          For SchNet: extract all fields ------------> nop, correct
        """
        raise NotImplementedError

    # -------------------------
    # Concrete Workflow Methods
    # -------------------------
    
    def preprocess(self):
        """Workflow for train/validation: Load -> Format -> Split."""
        dataset = self._load_dataset()
        processed = self._format_dataset(dataset, is_inference=False)
        return self.split(processed)

    def preprocess_test(self):
        """Workflow for test: Load -> Format -> Return."""
        dataset = self._load_test_dataset()
        return self._format_dataset(dataset, is_inference=False)

    def preprocess_inference(self):
        """Workflow for inference: Load -> Format (no target) -> Return."""
        dataset = self._load_dataset()
        return self._format_dataset(dataset, is_inference=True)



    # ---------------------------
    # Logging / artifacts
    # ---------------------------
    def get_artifacts(self):
        """Return all key artifacts and metadata for logging."""
        return {
            "val_ratio": self.val_ratio,
            "subset": self.subset,
            "target": self.target,
        }