

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












import mlflow
from mlflow.tracking import MlflowClient

def select_best_model(experiment_name, run_name='tuning', metric="val_mse", model_type=None, data_version="v1"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Build filter
    filter_str = f"run_name = '{run_name}' AND tags.data_version = '{data_version}'"
    if model_type:
        filter_str += f" AND tags.model_type = '{model_type}'"

    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric} ASC"], #Sorting ascending (ASC) ensures: lowest MSE → rank 1 (best)
    )

    if not runs:
        raise ValueError(f"No runs found for model_type='{model_type}' and data_version='{data_version}'.")

    # Take best run
    best_run = runs[0]
    run_id = best_run.info.run_id

    print(f"Best {model_type or 'overall'} run ID: {run_id}")
    print(f"Best {metric.upper()}: {best_run.data.metrics[metric]:.4f}")

    # Common part
    model_uri = f"runs:/{run_id}/model"


    model_type = best_run.data.tags.get("model_type")

    # return
    return {
        "run_id": run_id,
        "model_type": model_type,
        "model_uri": model_uri,
    }




from pathlib import Path
import mlflow



root_dir = Path(__file__).resolve().parents[1]

# Tracking DB
#mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}") OK

#mlflow.set_tracking_uri("http://mlflow:5000") -> with docker service

mlflow.set_tracking_uri("http://127.0.0.1:5000") # works locally




def main():




	res = select_best_model("qm9")

	model = mlflow.pytorch.load_model(res["model_uri"])
	print(model)



	# inference test

	ds = DummyDataset(root=None)
	sample = ds[0]      # get a single molecule

	# --- 2. Device ---
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# --- 4. Move sample to device ---
	sample = sample.to(device)


	# Create batch vector
	batch = torch.zeros(sample.z.size(0), dtype=torch.long, device=device)


	# --- 5. Predict ---
	with torch.no_grad():
	    pred = model(batch)

	print("Prediction:", pred)


if __name__=="__main__":
	main()