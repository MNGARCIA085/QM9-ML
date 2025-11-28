import torch


class ModelCheckpoint:
    def __init__(self, filepath, mode="min"):
        self.filepath = filepath
        self.best = float("inf") if mode == "min" else -float("inf")
        self.mode = mode

    def step(self, metric, model):
        improved = (
            metric < self.best if self.mode == "min"
            else metric > self.best
        )

        if improved:
            self.best = metric
            torch.save(model.state_dict(), self.filepath)
            return True
        return False