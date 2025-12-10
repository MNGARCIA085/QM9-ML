import subprocess
import os
import hydra
from omegaconf import DictConfig




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # List of preprocessor variants
    val_ratios = ".2"
    subsets = "1000,2000"

    # Tuning different models
    model_types = "mlp,gcn,schnet"

    epochs_trials = "5"

    epochs = "40"



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



# 4000, 40 y 400 da bárbaro (mismo con 200)

# 2000, 40 bte bien!!

# 2000, 40 pretty good


# 4000 y 50 bte bien

#6000, 50 y 400 buena comb.