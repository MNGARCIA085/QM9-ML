import torch

def get_plateau_scheduler(
    optimizer,
    mode="min",
    factor=0.7,
    patience=2,
    min_lr=1e-6
):
    """
    Returns a ReduceLROnPlateau scheduler from PyTorch built-ins.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )




# study them!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_cosine_scheduler(
    optimizer,
    T_max=50,
    eta_min=1e-6
):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
    )


def get_onecycle_scheduler(
    optimizer,
    max_lr,
    total_steps
):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps
    )
