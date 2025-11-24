import torch

def compute_metrics(y_true, y_pred):
	# Convert to CPU tensors for safety
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        # MSE
        mse = torch.mean((y_pred - y_true)**2).item()

        # RMSE
        rmse = torch.sqrt(torch.mean((y_pred - y_true)**2)).item()

        # MAE
        mae = torch.mean(torch.abs(y_pred - y_true)).item()

        # R2
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_mean)**2)
        ss_res = torch.sum((y_true - y_pred)**2)
        r2 = 1 - ss_res / ss_tot
        r2 = r2.item()

        # Explained variance
        var_y = torch.var(y_true, unbiased=False)
        ev = 1 - torch.var(y_true - y_pred, unbiased=False) / var_y
        ev = ev.item()

        # return
        return {
        	'mse': mse,
        	'rmse': rmse,
        	'mae': mae,
        	'r2': r2,
        	'ev': ev
        }