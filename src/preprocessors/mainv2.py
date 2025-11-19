from .first import MLPPreprocessor
from torch_geometric.loader import DataLoader


from .registry import PreprocessorRegistry


from src.preprocessors.mlp import MLPPreprocessor
from src.preprocessors.gcn import GCNPreprocessor
from src.preprocessors.schnet import SchNetPreprocessor



def main():

	from pathlib import Path

	ROOT = Path(__file__).resolve().parents[2]     # project/
	DATA_DIR = ROOT / "data" / "QM9"


	print(DATA_DIR)


	model_type = 'gcn'


	prep = PreprocessorRegistry.create(
	    model_type,
	    target=0,
	    root=DATA_DIR,
	    subset=1000,
	)

	train_ds, val_ds = prep.preprocess()
	#python train.py model_type=gcn target=2

	train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=64)


	for t in train_loader:
	    print(t[1].y, t[1].z, t[1].edge_index)
	    print(t[1].y.shape, t[1].z.shape)
	    break




if __name__==main():
	main()
