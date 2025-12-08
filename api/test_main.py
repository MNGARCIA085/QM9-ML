import torch
import torch.nn as nn
from torch_geometric.nn import SchNet


class SchNetRegressor(nn.Module):
    def __init__(
        self,
        hidden_channels=256,
	    num_filters=128,
	    num_interactions=6,  # inferred from interactions.0 ... interactions.5
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

        self.regressor = nn.Identity()
        # self.regressor = nn.Linear(1,1 )

    def forward(self, z, pos, batch):
        return self.schnet(z=z, pos=pos, batch=batch)  # [N,1]



def load_model():
    model = SchNetRegressor(
        hidden_channels=256,
	    num_filters=128,
	    num_interactions=6,  # inferred from interactions.0 ... interactions.5
	    num_gaussians=50,
	    cutoff=5.0,
	    readout="add"
    )
    
    state_dict = torch.load(
        "model/model.pt",
        map_location="cpu",
        weights_only=True  # optional
    )
    
    # to device
    model.load_state_dict(state_dict)
    model.eval()
    return model









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













def main():
	
	#print(model)



	import torch

	# --- 1. Load dataset ---
	ds = DummyDataset(root=None)
	sample = ds[0]      # get a single molecule

	# --- 2. Device ---
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# --- 3. Load model ---
	model = load_model()

	# --- 4. Move sample to device ---
	sample = sample.to(device)


	# Create batch vector
	batch = torch.zeros(sample.z.size(0), dtype=torch.long, device=device)


	# --- 5. Predict ---
	with torch.no_grad():
	    pred = model(sample.z, sample.pos, None)#batch)

	print("Prediction:", pred)




if __name__=="__main__":
	main()