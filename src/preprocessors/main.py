from .first import MLPPreprocessor
from torch_geometric.loader import DataLoader




def main():

	from pathlib import Path

	ROOT = Path(__file__).resolve().parents[2]     # project/
	DATA_DIR = ROOT / "data" / "QM9"


	print(DATA_DIR)


	
	prep = MLPPreprocessor(root=DATA_DIR, subset=1000) # change to the appr. dir
	train_ds, val_ds = prep.preprocess()

	train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=64)


	for t in train_loader:
	    print(t[1].y, t[1].z, t[1].edge_index)
	    print(t[1].y.shape, t[1].z.shape)
	    break




if __name__==main():
	main()
