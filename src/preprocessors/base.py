import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]     # project/
DATA_DIR = ROOT / "data" / "QM9"
#DATA_DIR = ROOT / "data" / "QM9 / qm9_subset_1000.pt"



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
    def _load_dataset(self): # ok but memory intensive
        """Loads the dataset once and caches it."""
        if self._dataset is None:
            # Load the dataset (expensive I/O operation)
            dataset = self.dataset_cls(root=self.root, transform=self.transform)

            if self.subset:
                # Apply subset slicing
                dataset = dataset[:self.subset]
            
            self._dataset = dataset
            
        return self._dataset


    # alwasy load tiny version
    def _load_datasetv7(self):
        import os
        cache_path = os.path.join(self.root, f"qm9_subset_150.pt")


        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached QM9 subset ({self.subset}) from {cache_path}")
            self._dataset = torch.load(cache_path,  weights_only=False)
            return self._dataset

    # --------------------------------
    # Lazy-loading with subset caching
    # --------------------------------
    
    def _load_datasetOK(self):
        import os
        #Loads dataset or subset from disk if available.
        if self._dataset is not None:
            return self._dataset

        # -------------------------------
        # 1) If subset requested, check cache
        # -------------------------------
        if self.subset:
            cache_path = os.path.join(self.root, f"qm9_subset_{self.subset}.pt")

            if os.path.exists(cache_path):
                print(f"[INFO] Loading cached QM9 subset ({self.subset}) from {cache_path}")
                self._dataset = torch.load(cache_path,  weights_only=False)
                return self._dataset

        # -------------------------------
        # 2) Load full dataset (first time)
        # -------------------------------
        print("[INFO] Loading full QM9 dataset (first time)...")
        full_dataset = self.dataset_cls(root=self.root, transform=self.transform)

        # -------------------------------
        # 3) Apply subset slicing (if needed)
        # -------------------------------
        if self.subset:
            subset_ds = full_dataset[:self.subset]
            
            # Save the subset to avoid loading full QM9 next time
            print(f"[INFO] Saving QM9 subset ({self.subset}) to cache...")
            os.makedirs(self.root, exist_ok=True)
            torch.save(subset_ds, cache_path)

            self._dataset = subset_ds
        else:
            self._dataset = full_dataset

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
    # Concrete Workflow Methods (Handle Redundancy)
    # -------------------------
    
    def preprocess(self):
        """Workflow for train/validation: Load -> Format -> Split."""
        dataset = self._load_dataset()
        processed = self._format_dataset(dataset, is_inference=False)
        return self.split(processed)

    def preprocess_test(self):
        """Workflow for test: Load -> Format -> Return."""
        dataset = self._load_dataset()
        return self._format_dataset(dataset, is_inference=False)

    def preprocess_inference(self):
        """Workflow for inference: Load -> Format (no target) -> Return."""
        dataset = self._load_dataset()
        return self._format_dataset(dataset, is_inference=True)