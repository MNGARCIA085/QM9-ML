import torch
from torch_geometric.data import Data
import pytest


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


"""




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





# tests/conftest.py
import torch
from torch_geometric.data import Data

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
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]








"""
after prep

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

