import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class SimpleMLP(nn.Module):
    def __init__(self, num_atom_types=100, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(num_atom_types, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Store hyperparameters for MLflow + API reproducibility
        self.config = dict(
            hidden=hidden,
            num_atom_types=num_atom_types,
        )
    
    def forward(self, batch):
        x = self.emb(batch.z)
        x = global_mean_pool(x, batch.batch)
        return self.fc(x).view(-1)