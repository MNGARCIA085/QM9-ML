


from torch_geometric.loader import DataLoader
from src.preprocessors.registry import PreprocessorRegistry
from src.preprocessors.mlp import MLPPreprocessor
from src.preprocessors.gcn import GCNPreprocessor
from src.preprocessors.schnet import SchNetPreprocessor



def loaders():

	from pathlib import Path

	ROOT = Path(__file__).resolve().parents[2]     # project/
	DATA_DIR = ROOT / "data" / "QM9"


	print(DATA_DIR)


	model_type = 'schnet'


	prep = PreprocessorRegistry.create(
	    model_type,
	    target=0,
	    root=DATA_DIR,
	    subset=1000,
	)

	train_ds, val_ds = prep.preprocess()
	train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=64)
	return train_loader, val_loader






import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, GCNConv, global_mean_pool
from torch import nn







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3️⃣ Training / evaluation function ---
def run_epoch(loader, model, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
        out = model(batch.z, batch.pos, batch.batch)  # [num_graphs, hidden_channels]


        #print(out.shape)
        
        #pred = regressor(out).squeeze(-1)             # [num_graphs]
        pred = out.squeeze(-1)

        #print(pred.shape)
        
        target = batch.y.squeeze(-1)                  # [num_graphs]


        #print(target.shape)
        
        loss = criterion(pred, target)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)





def main():


	# loaders
	train_loader, val_loader = loaders()


	# --- 2️⃣ Define SchNet model ---
	model = SchNet(
	    hidden_channels=64,
	    num_filters=64,
	    num_interactions=3,
	    num_gaussians=50,
	    cutoff=10.0,
	    readout='add'  # important: aggregates node embeddings to graph embeddings
	)
	regressor = torch.nn.Linear(64, 1)

	optimizer = Adam(list(model.parameters()) + list(regressor.parameters()), lr=1e-3)
	criterion = MSELoss()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model, regressor = model.to(device), regressor.to(device)

	# --- 3️⃣ Training / evaluation function ---
	def run_epoch(loader, train=True):
	    model.train() if train else model.eval()
	    total_loss = 0
	    for batch in loader:
	        batch = batch.to(device)
	        if train:
	            optimizer.zero_grad()
	        out = model(batch.z, batch.pos, batch.batch)  # [num_graphs, hidden_channels]


	        #print(out.shape)
	        
	        #pred = regressor(out).squeeze(-1)             # [num_graphs]
	        pred = out.squeeze(-1)

	        #print(pred.shape)
	        
	        target = batch.y.squeeze(-1)                  # [num_graphs]


	        #print(target.shape)
	        
	        loss = criterion(pred, target)
	        if train:
	            loss.backward()
	            optimizer.step()
	        total_loss += loss.item() * batch.num_graphs
	    return total_loss / len(loader.dataset)

	# --- 4️⃣ Training loop ---
	for epoch in range(1, 11):
	    train_loss = run_epoch(train_loader, train=True)
	    val_loss = run_epoch(val_loader, train=False)
	    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")



if __name__==main():
	main()