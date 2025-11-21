# mlp_tuner.py
from .base import BaseTuner
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

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

class MLPTuner(BaseTuner):
    def create_model(self, trial):
        hidden = trial.suggest_categorical("hidden", [32, 64, 128])
        return SimpleMLP(hidden=hidden).to(self.device)

    def objective(self, trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)

        model = self.create_model(trial)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(5):  # small epochs for quick test
            self.run_epoch(train_loader, model, criterion, optimizer)
        val_loss = self.run_epoch(val_loader, model, criterion)
        return val_loss
