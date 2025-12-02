import torch
import torch.nn as nn
import torch.optim as optim
import optuna



class BaseTuner:

    trainer_cls = None   # subclasses must define this

    def __init__(self, train_ds, val_ds, epochs_trials=5, device=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs_trials = epochs_trials # small epochs for quick test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None


    def _make_trainer(self):
        return self.trainer_cls(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            device=self.device,
        )


    # objective
    def objective(self, trial, **kwargs):
        raise NotImplementedError("Subclasses must implement objective()")

    # tune
    def tune(self, n_trials=10, **kwargs):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, **kwargs), # to pass kwargs
                       n_trials=n_trials)


        print("Best params:", study.best_params)
        best_attrs = study.best_trial.user_attrs # metrics are a key under this
        print("All metrics:", best_attrs)

        
        # store for later (MLflow, logging, checkpoints…)
        self.best_params = study.best_params
        self.best_attrs  = best_attrs   # metrics / extras (right now i just for my best trial for a few epochs)

        # test
        
        rows = []
        for t in study.trials:
            row = {}
            row.update(t.params)
            row.update(t.user_attrs)
            row["value"] = t.value
            row["number"] = t.number
            rows.append(row)
        print(rows)
        trials_data = rows



        # return
        return study.best_params, best_attrs, trials_data
    




    



# inherirance + comnposition
# https://gemini.google.com/app/9feb954ea942a21a?hl=es