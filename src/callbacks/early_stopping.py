
class EarlyStopping:
    def __init__(self, patience=20, mode="min", monitor="val_loss"):
        self.patience = patience
        self.mode = mode
        self.monitor = monitor
        self.best = float("inf") if mode == "min" else -float("inf")
        self.wait = 0
        self.stop_training = False

    def step(self, metric_value):
        better = (
            metric_value < self.best if self.mode == "min"
            else metric_value > self.best
        )

        if better:
            self.best = metric_value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stop_training = True
