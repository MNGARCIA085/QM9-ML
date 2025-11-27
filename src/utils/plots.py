import matplotlib.pyplot as plt
import os

#from hydra.utils import to_absolute_path


def plot_losses(train_values, val_values, plot_name='Loss Curve', ylabel='Loss (MSE)', xlabel="Epoch"):
    """
    Plot training and validation curves (it can be for loss or accs)
    
    Args:
        train_values (list or array): Training values per epoch.
        val_values (list or array): Validation values per epoch.
        ylabel (str): Label for y-axis
        xlabel (str): Label for x-axis (default "Epoch").
        plot_name (str): Acc or Loss.
        
    Returns:
        str: The path where the plot was saved.
    """
    plt.figure()
    plt.plot(train_values, label=f"Train {ylabel}")
    plt.plot(val_values, label=f"Validation {ylabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(os.getcwd(), plot_name)
    plt.savefig(save_path)
    plt.close()
    return save_path