from types import SimpleNamespace

import torch

from utils.lr_helper import get_scheduler


def test_get_scheduler_builds_cosine_annealing_lr():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    config = SimpleNamespace(
        type="CosineAnnealingLR",
        kwargs={"T_max": 1000, "eta_min": 0.0},
    )

    scheduler = get_scheduler(optimizer, config)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
