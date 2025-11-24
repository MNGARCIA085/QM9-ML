import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

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