import torch.nn as nn
from torch_geometric.nn import SchNet


class SchNetRegressor(nn.Module):
    def __init__(
        self,
        hidden_channels=64,
        num_filters=64,
        num_interactions=3,
        num_gaussians=50,
        cutoff=5.0,
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

        self.regressor = nn.Identity()  # <--- Important, same as nn.Linear(1, 1)?

    def forward(self, z, pos, batch):
        return self.schnet(z=z, pos=pos, batch=batch)  # [N,1]





