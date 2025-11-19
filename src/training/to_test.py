


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


	model_type = 'mlp'


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



def run_epoch(loader, model, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        #print(batch.z.shape)
        #print(out.shape)

        pred = out.view(-1)
        target = batch.y.view(-1)

        loss = criterion(pred, target)

        #print("pred", pred.shape, "target", target.shape, "num_graphs", batch.num_graphs)


        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)



class SimpleMLP(nn.Module):
    def __init__(self, num_atom_types=100, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(num_atom_types, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, batch):
        x = self.emb(batch.z)       # [num_nodes, hidden]
        x = global_mean_pool(x, batch.batch)  # [num_graphs, hidden]
        out = self.fc(x)            # [num_graphs, 1]
        return out




def main():
	model_mlp = SimpleMLP().to(device)
	optimizer = Adam(model_mlp.parameters(), lr=1e-3)
	criterion = MSELoss()



	# loaders
	train_loader, val_loader = loaders()


	# Train for a few epochs
	for epoch in range(3):
	    train_loss = run_epoch(train_loader, model_mlp, criterion, optimizer)
	    val_loss = run_epoch(val_loader, model_mlp, criterion)
	    print(f"[MLP] Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")



if __name__==main():
	print('nico')
	main()