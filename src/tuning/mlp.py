import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import global_mean_pool
from ray import tune

from .base import BaseTuner


# ---------------- Simple PYG MLP model ---------------- #
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


# ---------------- Tuner ---------------- #
class MLPTuner(BaseTuner):

    def __init__(self, train_ds, val_ds):
        super().__init__(train_ds, val_ds)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # One epoch
    def run_epoch(self, loader, model, criterion, optimizer=None):
        training = optimizer is not None
        model.train() if training else model.eval()

        total_loss = 0
        total_graphs = 0

        for batch in loader:
            batch = batch.to(self.device)
            pred = model(batch)
            target = batch.y.view(-1)
            loss = criterion(pred, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        return total_loss / total_graphs

    # Ray training function
    def _train_model_ray(self, config):

        train_loader, val_loader = self.create_loaders(
            batch_size=config["batch_size"]
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SimpleMLP(
            num_atom_types=100,
            hidden=config["hidden"],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()

        for epoch in range(config["epochs"]):
            train_loss = self.run_epoch(train_loader, model, criterion, optimizer)
            val_loss = self.run_epoch(val_loader, model, criterion)

            tune.report({"val_loss": val_loss})

    def get_tune_config(self):
        return {
            "hidden": tune.choice([32, 64, 128]),
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32]),
            "epochs": 5,
        }
