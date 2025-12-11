## QM9


### Project Overview

This project predicts the molecular dipole moment using the QM9 benchmark dataset. The goal is not to reach state-of-the-art accuracy, but to demonstrate how to structure and execute a real machine learning project end-to-end.

The focus is on good engineering practices: a clear folder structure, reproducible experiments, Dockerized environments, hyperparameter tuning, model comparison, metric tracking, and clean separation between training code and deployment code. The trained model will be consumed by a separate API repository, where it is served for real-time predictions.

The aim is to show practical ML engineering skills rather than maximizing performance.



### Setup

#### 1. Create a virtual environment and install dependencies

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


#### 2. Run the main pipeline

```bash
python -m scripts.pipeline
```

#### 3. Run tests

```bash
pytest
```

### Docker usage

#### 1. Build the container

```bash
docker build -f Dockerfile -t ml_qm9:latest .
```

#### 2. Run the container

```bash
docker run -it ml_qm9:latest
```


### Running individual scripts

You can execute any module in the scripts/ directory directly and override parameters from the command line:


```bash
python -m scripts.tuning model_type=schnet shared.epochs=50
python -m scripts.tuning model_type=gcn preprocessor.val_ratio=0.1 shared.num_trials=10
python -m scripts.evaluation
```


### Citation

Dataset:

Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014).
Quantum chemistry structures and properties of 134 kilo molecules.
Scientific Data, 1, 140022.

PyTorch Geometric dataset loader:

Fey, M., & Lenssen, J. E. (2019).
Fast Graph Representation Learning with PyTorch Geometric.
arXiv:1903.02428.


