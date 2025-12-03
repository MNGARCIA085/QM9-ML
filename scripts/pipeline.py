import subprocess
import os
import hydra
from omegaconf import DictConfig




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # List of preprocessor variants
    val_ratios = ".2"
    subsets = "12000"

    # Tuning different models
    model_types = "mlp,gcn,schnet"

    epochs_trials = "5, 8"

    epochs = "100"



    cmds = [
        "python",
        "-m",
        "scripts.tuning",
        "-m", # --multirun
        f"model_type={model_types}",
        f"preprocessor.val_ratio={val_ratios}",
        f"preprocessor.subset={subsets}",
        f"shared.epochs_trials={epochs_trials}",
        f"shared.epochs={epochs}",
    ]
    
    # Pass the list
    subprocess.run(cmds, check=True)



    # call evaluate script
    cmds_eval = [
        "python",
        "-m",
        "scripts.evaluation",
    ]
    
    # Pass the list
    subprocess.run(cmds_eval, check=True)




if __name__ == "__main__":
    main()

