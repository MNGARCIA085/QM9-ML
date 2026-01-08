import torch
import torch.nn as nn
from torch_geometric.data import Data
import pytest


# data example
@pytest.fixture
def qm9_like_sample():
    num_nodes = 5
    num_edges = 8
    num_targets = 19

    return Data(
        x=torch.randn(num_nodes, 11),
        pos=torch.randn(num_nodes, 3),
        z=torch.randint(1, 10, (num_nodes,)), #1-D tensor with length = num_nodes;z=[12]→ shape torch.Size([12])
                                              # z is the atomic number of each atom in the molecule (cehck!!!!)
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 4),
        y=torch.randn(1, num_targets),   # REAL QM9: graph-level y
        smiles="[H]C([H])([H])[H]",
        name="gdb_test",
        idx=torch.tensor([1])
    )



# dummy dataset
class DummyDataset:
    """Dataset that mimics QM9 behavior."""
    def __init__(self, root, transform=None):
        self.data_list = [
            Data(
                x=torch.randn(5, 11),
                edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]]),
                edge_attr=torch.randn(4, 4),
                y=torch.randn(1, 19),
                pos=torch.randn(5, 3),
                z=torch.randint(1, 10, (5,)),
                smiles="H2O",
                name="mol1",
                idx=torch.tensor([0])
            ),
            Data(
                x=torch.randn(7, 11),
                edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]]),
                edge_attr=torch.randn(4, 4),
                y=torch.randn(1, 19),
                pos=torch.randn(7, 3),
                z=torch.randint(1, 10, (7,)),
                smiles="CO2",
                name="mol2",
                idx=torch.tensor([1])
            ),
            Data(
                x=torch.randn(7, 11),
                edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]]),
                edge_attr=torch.randn(4, 4),
                y=torch.randn(1, 19),
                pos=torch.randn(7, 3),
                z=torch.randint(1, 10, (7,)),
                smiles="CO2",
                name="mol2",
                idx=torch.tensor([1])
            ),
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]







# ---------------------------------------------------------
#                       MLP
# ---------------------------------------------------------
def make_dataset_mlp(num_graphs=5, hidden=8):
    """
    Creates a dataset of simple Data objects for the MLP.
    Each graph has x as a single graph-level feature vector.
    """
    dataset = []
    for _ in range(num_graphs):
        x = torch.rand(1, hidden)             # MLP takes graph-level features
        y = torch.rand(1)                     # regression target
        dataset.append(Data(x=x, y=y))
    return dataset


@pytest.fixture
def dataset_mlp():
    def _make(num_graphs=5, hidden=8): # to be able to pass params
        return make_dataset_mlp(num_graphs, hidden)
    return _make



# A simple mock MLP: mean over features → linear layer → scalar output
class MockMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        # data.x: [1, hidden]
        return self.lin(data.x).view(-1)  # returns [1]


@pytest.fixture
def mock_mlp():
    return MockMLP(hidden=8)





# ---------------------------------------------------------
#                       GCN
# ---------------------------------------------------------
def make_dataset_gcn(num_graphs=5, hidden=16):
    """
    Creates a list of real PyG Data objects so DataLoader works correctly.
    """
    dataset = []
    for _ in range(num_graphs):
        x = torch.rand(10, hidden)               # node features
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # trivial edges
        y = torch.rand(1)                        # graph-level target
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


@pytest.fixture
def dataset_gcn():
    def _make(num_graphs=5, hidden=16): # to be able to pass params
        return make_dataset_gcn(num_graphs, hidden)
    return _make


# Mock model (same idea as SimpleGCN, but extremely minimal)
class MockGCN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data,  *args, **kwargs):
        # simple graph-level prediction = mean over node features
        out = self.lin(data.x).mean(dim=0, keepdim=True)  # [1,1]
        return out

@pytest.fixture
def mock_gcn():
    # mock_gcn is the object returned by the fixture
    return MockGCN(hidden=16)




# ---------------------------------------------------------
#                       SCHNET
# ---------------------------------------------------------
def make_dataset_schnet(num_samples=20, num_nodes=5):
    dataset = []
    for _ in range(num_samples):
        pos = torch.randn(num_nodes, 3)          # Positions
        z = torch.randint(1, 10, (num_nodes,))   # Atomic numbers
        y = torch.randn(1)                       # Scalar property

        # SchNet does NOT require edge_index explicitly
        data = Data(pos=pos, z=z, y=y, num_nodes=num_nodes)
        dataset.append(data)

    return dataset

@pytest.fixture
def dataset_schnet():
    def _make(num_samples=20, num_nodes=5): # to be able to pass params
        return make_dataset_schnet(num_samples, num_nodes)
    return _make





# remeber -> Fixtures are values, not constructors.


"""

Original data example


Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3], z=[5], smiles='[H]C([H])([H])[H]', name='gdb_1', idx=[1])

Data(
    x=[N, 11],
    pos=[N, 3],
    z=[N],
    edge_index=[2, E],
    edge_attr=[E, 4],
    y=[1, 19],           <-- IMPORTANT: global targets, NOT per-node
    smiles=...,
    name=...,
    idx=[1]
)


-> after prep

Data(
    x=[num_nodes, num_node_features]   # QM9 uses 11 features → x=[N,11]
    edge_index=[2, num_edges]
    edge_attr=[num_edges, 4]
    y=[1,1]                            # after your preprocessing
    pos=[N,3]
    z=[N]
    smiles=str
    name=str
    idx=[1]
)


Data(x=[12, 11], edge_index=[2, 24], edge_attr=[24, 4], y=[1, 1], pos=[12, 3], z=[12], 
smiles='[H]C#C[C@]1([H])OC([H])([H])C1([H])[H]', name='gdb_552', idx=[1])
"""

