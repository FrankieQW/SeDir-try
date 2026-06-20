import torch


def get_scheduler(optimizer, config):
    if config.type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.kwargs)
    if config.type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config.kwargs)
    else:
        raise NotImplementedError
