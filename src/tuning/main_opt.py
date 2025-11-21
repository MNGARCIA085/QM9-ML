import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, SchNet
import optuna

# ---------------- Models ---------------- #

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
        x = self.emb(batch.z)
        x = global_mean_pool(x, batch.batch)
        return self.fc(x).view(-1)

class SimpleGCN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, batch):
        x, edge_index = batch.x.float(), batch.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch.batch)
        return self.fc(x).view(-1)

# ---------------- Tuner ---------------- #

class GraphTuner:
    def __init__(self, train_ds, val_ds, model_type="mlp"):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type.lower()

    def create_model(self, trial):
        if self.model_type == "mlp":
            hidden = trial.suggest_categorical("hidden", [32, 64, 128])
            return SimpleMLP(hidden=hidden).to(self.device)
        elif self.model_type == "gcn":
            hidden = trial.suggest_categorical("hidden", [32, 64, 128])
            return SimpleGCN(hidden=hidden).to(self.device)
        elif self.model_type == "schnet":
            # Using default SchNet settings; can be tuned further
            return SchNet(hidden_channels=64, num_filters=64, num_interactions=3).to(self.device)
        else:
            raise ValueError(f"Unknown model_type {self.model_type}")

    def run_epoch(self, loader, model, criterion, optimizer=None):
        model.train() if optimizer else model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1).float())
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    def objective(self, trial):
        # hyperparameters
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)

        model = self.create_model(trial)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # short training for quick tests
        epochs = 5
        for _ in range(epochs):
            self.run_epoch(train_loader, model, criterion, optimizer)
        val_loss = self.run_epoch(val_loader, model, criterion)
        return val_loss

    def tune(self, n_trials=5):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        return study.best_params





from src.preprocessors.mlp import MLPPreprocessor



def main():
    prep = MLPPreprocessor(subset=1000)
    train_ds, val_ds = prep.preprocess()
    # train_ds, val_ds are PyG datasets
    tuner = GraphTuner(train_ds, val_ds, model_type="mlp")  # or "gcn", "schnet"
    best_config = tuner.tune(n_trials=5)



if __name__=="__main__":

    main()