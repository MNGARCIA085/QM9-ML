


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





	model_type = 'gcn'
	# args -> depending on the model


	prep = PreprocessorRegistry.create(
	    model_type,
	    target=0,
	    root=DATA_DIR,
	    subset=1000,
	    # kwargs
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

# --- 2️⃣ Define Graph CNN ---
class SimpleGCN(nn.Module):
    def __init__(self, hidden=64, num_atom_types=100):
        super().__init__()
        self.emb = nn.Embedding(num_atom_types, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, batch):
        x = self.emb(batch.z)           # [num_nodes, hidden]
        x = self.conv1(x, batch.edge_index) # using edge_index here!!!!!!!!!!!!!!!!!!!!
        x = torch.relu(x)
        x = self.conv2(x, batch.edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch.batch)  # [num_graphs, hidden]
        out = self.fc(x)                        # [num_graphs, 1]
        return out



# --- 3️⃣ Training / evaluation loop ---
def run_epoch(loader, model, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.squeeze(-1)
        target = batch.y.squeeze(-1)
        loss = criterion(pred, target)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)






def main():
	model_gcn = SimpleGCN().to(device)
	optimizer = Adam(model_gcn.parameters(), lr=1e-3)
	criterion = MSELoss()



	# loaders
	train_loader, val_loader = loaders()


	# --- 4️⃣ Training ---
	for epoch in range(1, 11):
	    train_loss = run_epoch(train_loader, model_gcn, criterion, optimizer)
	    val_loss = run_epoch(val_loader, model_gcn, criterion)
	    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")



if __name__==main():
	main()