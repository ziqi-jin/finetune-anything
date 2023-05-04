# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/optim/lr_scheduler.py  # noqa
# and https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/solver/lr_scheduler.py

from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import LRScheduler


AVAI_SCH = ["single_step", "multi_step", "warmup_multi_step", "cosine", "linear"]


def build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=1,
    gamma=0.1,
    warmup_factor=0.01,
    warmup_steps=10,
    max_epoch=1,
    n_epochs_init=50,
    n_epochs_decay=50,

):
    """A function wrapper for building a learning rate scheduler.
    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is
            single_step.
        stepsize (int or list, optional): step size to decay learning rate.
            When ``lr_scheduler`` is "single_step", ``stepsize`` should be an integer.
            When ``lr_scheduler`` is "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.
    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                "be an integer, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "warmup_multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(
                "For warmup multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=stepsize,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_steps,
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    elif lr_scheduler == "linear":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - n_epochs_init) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule
        )

    return scheduler


class WarmupMultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        **kwargs,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
