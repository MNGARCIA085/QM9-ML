import mlflow
import os
from mlflow.tracking import MlflowClient
from pathlib import Path
from .plots import plot_losses
import pandas as pd
import uuid


# Project root (2 levels up from this file)

root_dir = Path(__file__).resolve().parents[2]

# Tracking DB
mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}")

# Artifacts folder
artifact_dir = root_dir / "mlruns"
os.makedirs(artifact_dir, exist_ok=True)





# logging (for the tuning exps)
def logging(exp_name, run_name, artifacts, results, model_type, trials_data):

    # ensures artifact path is set
    mlflow.set_experiment(exp_name)

    # run
    with mlflow.start_run(run_name=run_name):

        # Tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("data_version", "v1") # hardcoded for now
        #mlflow.set_tag("data_version", artifacts["dvc_version"]) 


        # Common params
        mlflow.log_param("val_ratio", artifacts["val_ratio"])
        mlflow.log_param("subset", artifacts["subset"])
        mlflow.log_param("target", str(artifacts["target"]))

        # Metrics
        for name, value in results["val"]["metrics"].items():
            mlflow.log_metric(f"val_{name}", value)

        # Model (is always a torch model)        
        mlflow.pytorch.log_model(results["model"], name="model") # artifact_path
        
        # training curves
        loss_path = plot_losses(results["train"]["losses"], results["val"]["losses"], "loss_curve.png", 'Loss')
        mlflow.log_artifact(loss_path)
        os.remove(loss_path)
 
        # hyperparams (differnt dict depending on the model)
        mlflow.log_params(results["hyperparams"]) # results.hyperparams if i use a datatype

        #----------trials--------------
        # Save to a temporary JSON
        df = pd.DataFrame(trials_data)
        path = f"optuna_trials_{uuid.uuid4().hex}.json" # maybe name from logging config
        df.to_json(path, orient="records", indent=2)
        mlflow.log_artifact(path)
        os.remove(path)
        



"""

https://chatgpt.com/c/692e2037-86d0-8330-96dc-c7d7ba943843

from hydra.core.hydra_config import HydraConfig

run_dir = HydraConfig.get().run.dir
path = f"{run_dir}/optuna_trials.csv"
df.to_csv(path, index=False)
mlflow.log_artifact(path)
os.remove(path)

"""


"""
-------------------------------------
Saving







---------------------------------------
Loading

import mlflow

client = mlflow.tracking.MlflowClient()
run_id = "<run_id>"

# list artifacts in the "optuna" folder
artifacts = client.list_artifacts(run_id, path="optuna")
for a in artifacts:
    print(a.path)



local_path = client.download_artifacts(run_id, artifacts[0].path)

with open(local_path) as f:
    data = json.load(f)

"""





# get best model data
def select_best_model(experiment_name, run_name='tuning', metric="val_mse", model_type=None, data_version="v1"):
    """
    Select the best model overall or the best one of a specific type.
    
    Args:
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name of the run.
        metric (str): Metric used for ranking (default: 'mse').
        model_type (str, optional): Filter by model type (e.g., 'mlp' or 'schnet').
        data_version (str): Version tag of the data (default: 'v1').

    Returns:
        dict: Paths and metadata of the best run.
    """

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



"""
note: if i use metric for which highr is better: r2, ex:

direction = "ASC" if metric in ["mse", "rmse", "mae"] else "DESC"
insteado of order_by

"""






# log test results
def log_test_results(exp_name, tuning_run_id, model_type, metrics):

    # ensures artifact path is set
    mlflow.set_experiment(exp_name)
    
    # run
    with mlflow.start_run(run_name="test_evaluation"):

        # tags
        mlflow.set_tag("tuning_run_id", tuning_run_id)
        mlflow.set_tag("dataset", "test")
        mlflow.set_tag("data_version", "v1")

        mlflow.log_param("model_type", model_type)

        # Metrics
        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)








# analyze perfoemce when i move only one param



"""
# Delete the SQLite file
rm mlflow.db

# Delete all artifacts (by default in ./mlruns/)
rm -rf mlruns/

# Start MLflow server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000



# Start MLflow server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000


"""
