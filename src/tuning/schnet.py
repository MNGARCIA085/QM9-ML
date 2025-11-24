
# schnet_tuner.py
from .base import BaseTuner
from torch_geometric.nn import SchNet
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn as nn


from src.models.schnet import SchNetRegressor


from src.utils.metrics import compute_metrics


import torch


class SchNetTuner(BaseTuner):
    def __init__(self, train_ds, val_ds, epochs=5, device=None, **kwargs):
        super().__init__(train_ds, val_ds, epochs=epochs, device=device)



    # create model
    def create_model_from_params(self, params):
        return SchNetRegressor(hidden_channels=params["hidden_channels"],
        				num_filters=params["num_filters"],
        				num_interactions=params["num_interactions"]).to(self.device) # later cutoff





    # --- 3️⃣ Training / evaluation function ---
    # at leas the signatur goes in the base class
    def run_epoch(self, train, loader, model, criterion, optimizer=None): #train=True for training else False




        model.train() if train else model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            if train:
                optimizer.zero_grad()
            out = model(batch.z, batch.pos, batch.batch)  # [num_graphs, hidden_channels]


            #print(out.shape)
            
            #pred = regressor(out).squeeze(-1)             # [num_graphs]
            pred = out.squeeze(-1)

            #print(pred.shape)
            
            target = batch.y.squeeze(-1)                  # [num_graphs]


            #print(target.shape)
            
            loss = criterion(pred, target)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)



    # ---------------------------------------------------------
    # Tuning with Optuna
    # ---------------------------------------------------------
    
    # not that common to all subclasses
    def create_model(self, trial):
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64])
        num_filters = trial.suggest_categorical("num_filters", [32, 64])
        num_interactions = trial.suggest_int("num_interactions", 1, 5)
        return SchNet(hidden_channels=hidden_channels,
                      num_filters=num_filters,
                      num_interactions=num_interactions).to(self.device)


    # note it might be common to all subclasses!!!! (get preds may be diff an the attrs that i will store)
    def objective(self, trial):

	    batch_size = trial.suggest_categorical("batch_size", [16, 32])
	    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)

	    train_loader, val_loader = self.create_loaders(batch_size)

	    model = self.create_model(trial)
	    optimizer = optim.Adam(model.parameters(), lr=lr)
	    criterion = nn.MSELoss()

	    # 📌 Add scheduler HERE; change location later
	    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	        optimizer,
	        mode="min",
	        factor=0.7,
	        patience=2,
	        min_lr=1e-6
	    )

	    for epoch in range(5):
	        self.run_epoch(True, train_loader, model, criterion, optimizer)
	        val_loss = self.run_epoch(False, val_loader, model, criterion)

	        # 📌 Important: Call scheduler AFTER validation
	        scheduler.step(val_loss)

	        #print(f"Epoch {epoch}: lr = {optimizer.param_groups[0]['lr']}")

	    # final validation loss
	    val_loss = self.run_epoch(False, val_loader, model, criterion)

	    # ---- compute additional metrics ----
	    y_true, y_pred = self.get_predictions(val_loader, model)

	    # metrics
	    metrics = compute_metrics(y_true, y_pred)

	    # ---- store metadata in the trial ---- (later a dataclass maybe)
	    trial.set_user_attr("metrics", metrics)

	    return val_loss




    # make preds -> to check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def get_predictions(self, loader, model):
	    model.eval()

	    preds, trues = [], []

	    with torch.no_grad():
	        for batch in loader:
	            batch = batch.to(self.device)

	            y_hat = model(batch.z, batch.pos, batch.batch).squeeze(-1) # ipt. for shapes
	            y = batch.y.squeeze(-1).float()

	            preds.append(y_hat.cpu())
	            trues.append(y.cpu())

	    preds = torch.cat(preds)
	    trues = torch.cat(trues)
	    return trues, preds


