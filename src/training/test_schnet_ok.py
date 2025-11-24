import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.nn import SchNet







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
        cutoff=2.0,
    )

    train_ds, val_ds = prep.preprocess()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)


    for t in train_ds:
        print(t)
        break

    for t in train_loader:
        print(t)
        break

    return train_loader, val_loader









import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.nn import SchNet


class SchNetRegressorv0(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        num_filters=64,
        num_interactions=3,
        num_gaussians=50,
        cutoff=10.0,
        num_gaussians_embedding=50,
        readout="add",
    ):
        super().__init__()

        # Base SchNet
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians_embedding,
            cutoff=cutoff,
            readout=readout
        )

        # Your regressor
        self.regressor = nn.Linear(hidden_channels, 1)

    def forward(self, z, pos, batch):
        x = self.schnet(z=z, pos=pos, batch=batch)  # [num_graphs, hidden_channels]
        out = self.regressor(x)                     # [num_graphs, 1]
        return out




class SchNetRegressor(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        num_filters=64,
        num_interactions=3,
        num_gaussians=50,
        cutoff=10.0,
        readout="add"
    ):
        super().__init__()

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout
        )

        self.regressor = nn.Identity()  # <--- Important

    def forward(self, z, pos, batch):
        return self.schnet(z=z, pos=pos, batch=batch)  # [N,1]





def build_schnet_model(lr=1e-3):
    """Utility builder function."""
    model = SchNetRegressor()
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, optimizer, criterion, device





def main():

    model, optimizer, criterion, device = build_schnet_model()

    train_loader, val_loader = loaders()

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


if __name__=="__main__":
    main()