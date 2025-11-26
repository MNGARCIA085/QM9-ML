import mlflow
import os
from mlflow.tracking import MlflowClient
from pathlib import Path
#from .plots import plot_cm, plot_roc, plot_train_val


# Project root (2 levels up from this file)
root_dir = Path(__file__).resolve().parents[2]

# Tracking DB
mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}")

# Artifacts folder
artifact_dir = root_dir / "mlruns"
os.makedirs(artifact_dir, exist_ok=True)





# logging (for the tuning exps)
def logging(exp_name, run_name, artifacts, results, model_type):

    # ensures artifact path is set
    mlflow.set_experiment(exp_name)

    # run
    with mlflow.start_run(run_name=run_name):

        print(results)


        # Tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("data_version", "v1") # hardcoded for now
        #mlflow.set_tag("data_version", artifacts["dvc_version"]) 


        # Common params
        mlflow.log_param("val_ratio", artifacts["val_ratio"])
        mlflow.log_param("subset", artifacts["subset"])
        mlflow.log_param("target", str(artifacts["target"]))


        # Metrics (shared)
        #for m in ["mse", "rmse", "mae", "r2", "ev"]:
        #    mlflow.log_metric(m, getattr(results["val_metrics"], m)) # results.val.metrics
        #print(results)
        mlflow.log_metric("mse", results["val_metrics"]["mse"])


        # Model (is always a torch model)        
        mlflow.pytorch.log_model(results["model"], artifact_path="model")


        """
        # training curves
        loss_path = plot_train_val(results.train.losses, results.val.losses, "loss_curve.png", 'Loss')
        mlflow.log_artifact(loss_path)
        os.remove(loss_path)


        acc_path = plot_train_val(results.train.accs, results.val.accs, "acc_curve.png", 'Accuracy')
        mlflow.log_artifact(acc_path)
        os.remove(acc_path)  
        """

        # hyperparams (differnt dict depending on the model)
        mlflow.log_params(results["hyperparams"]) # results.hyperparams if i use a datatype








# get best model data -> aeqpt!!!!!!!!!!!!!!!!!!
def select_best_model(experiment_name, run_name='tuning', metric="mse", model_type=None, data_version="v1"):
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




# log test results -> adapt!!!!!!!!!!!!!!
def log_test_results(exp_name, tuning_run_id, model_type, results):

    # ensures artifact path is set
    mlflow.set_experiment(exp_name)
    
    # run
    with mlflow.start_run(run_name="test_evaluation"):

        # tags
        mlflow.set_tag("tuning_run_id", tuning_run_id)
        mlflow.set_tag("dataset", "test")
        mlflow.set_tag("data_version", "v1")

        mlflow.log_param("model_type", model_type)

        # Metrics (shared)
        for m in ["accuracy", "precision", "recall", "f1"]:
            mlflow.log_metric(m, getattr(results.metrics, m))

        # Plots
        cm_path = plot_cm(results.labels, results.preds)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        roc_path = plot_roc(results.labels, results.probs)
        mlflow.log_artifact(roc_path)
        os.remove(roc_path)









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
