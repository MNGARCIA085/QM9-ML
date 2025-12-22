import torch
from torch.utils.data import random_split
from pathlib import Path
from torch_geometric.datasets import QM9

DEFAULT_DATA_ROOT = Path.home() / ".cache" / "pyg_datasets" / "qm9"


class BasePreprocessor:
    """
    Base class for data preprocessing, handling common tasks like loading, 
    subsetting, splitting, and configuration.

    This class provides a simple and reproducible way to work with the QM9 dataset
    while keeping a fixed test split for final evaluation.

    The dataset is loaded once for training and hyperparameter tuning, where all
    samples except the last N are used. A fixed, reproducible test set is created 
    by reserving the last N samples after shuffling.  During training/tuning only 
    the training subset is loaded; the test subset is loaded separately and only 
    when needed for final evaluation.

    """
    def __init__(
        self,
        dataset_cls=QM9,
        root=None,
        transform=None,
        target=0,
        val_ratio=0.2,
        seed=42,
        last=400,
        subset=None,
    ):
        # Decide root directory
        if root is not None:
            self.root = Path(root)
        else:
            self.root = DEFAULT_DATA_ROOT

        # Ensure parent dir exists
        self.root.mkdir(parents=True, exist_ok=True)

        self.dataset_cls = dataset_cls
        self.transform = transform
        self.target = target
        self.val_ratio = val_ratio
        self.seed = 42 # hardcoded for safety (always get the same test set)
        self.subset = subset
        self.last = last # hardcoded for safety later, but adapt tests and all thats needed

        self._dataset = None
        self._test_dataset = None


    def _load_dataset(self):
        """Load train/val dataset with fixed, reproducible split."""
        if self._dataset is None:
            full = self.dataset_cls(
                root=str(self.root),
                transform=self.transform,
            )

            if self.last > 0:
                # this always give the same; same dataset, order, seed
                gen = torch.Generator().manual_seed(self.seed)
                
                """
                perm = torch.randperm(len(full), generator=gen)
                full = full[perm]
                train_part = full[:-self.last]
                """


                perm = torch.randperm(len(full), generator=gen).tolist()
                if self.last > 0:
                    train_indices = perm[:-self.last]
                else:
                    train_indices = perm
                train_part = torch.utils.data.Subset(full, train_indices)


            else:
                train_part = full

            # Optional subset for fast experiments
            if self.subset:
                train_part = train_part[:self.subset]

            self._dataset = train_part

        return self._dataset



    def _load_test_dataset(self):
        """Load fixed test dataset."""
        if self._test_dataset is None:
            full = self.dataset_cls(
                root=str(self.root),
                transform=self.transform,
            )

            if self.last == 0:
                raise ValueError("last must be > 0 to create a test set")

            # always returns a fixed holdout set
            gen = torch.Generator().manual_seed(self.seed)
            
            """
            perm = torch.randperm(len(full), generator=gen)
            full = full[perm]
            self._test_dataset = full[-self.last:]
            """
            perm = torch.randperm(len(full), generator=gen).tolist()
            test_indices = perm[-self.last:]
            self._test_dataset = torch.utils.data.Subset(full, test_indices)


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
          For SchNet: ....
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