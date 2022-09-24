import math
import torch

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
import numpy as np


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim,
        total_steps: int,
        warmup_factor: float = 0.001,
        warmup_pct: float = 0.05,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        """
        last_epoch : The index of the last batch.This parameter is used when resuming a training job. 
        this number represents the total number of batches computed, not the total number of epochs computed. 
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
        """
        self.base_lrs = None
        self.last_epoch = None
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_pct * total_steps)
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_steps,
            self.warmup_factor)
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr * warmup_factor * 0.5 *
            (1.0 + math.cos(math.pi * self.last_epoch / self.total_steps))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()

    def _get_warmup_factor_at_iter(self, method: str, iters: int,
                                   warmup_iters: int,
                                   warmup_factor: float) -> float:
        """
        Return the learning rate warmup factor at a specific iteration.
        See https://arxiv.org/abs/1706.02677 for more details.
        Args:
            method (str): warmup method; either "constant" or "linear".
            iters (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).
        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iters >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iters / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class yolo_lr_scheduler(object):
    def __init__(self, optimizer, epochs, hyp, nb):
        self.optimizer = optimizer
        self.epochs = epochs
        self.hyp = hyp
        self.lr0 = hyp["lr0"]
        self.lrf = hyp["lrf"]  # final OneCycleLR learning rate (lr0 * lrf)
        self.momentum = hyp["momentum"]
        self.weight_decay = hyp["weight_decay"]
        self.warmup_epochs = int(hyp["warmup_epochs"])
        self.warmup_momentum = hyp["warmup_momentum"]
        self.warmup_bias_lr = hyp["warmup_bias_lr"]
        self.warmup_steps = self.warmup_epochs * nb
        self.one_cycle_lr_scheduler = self.one_cycle_lr_scheduler(
            hyp, self.epochs - self.warmup_epochs, optimizer)
        self.step = 0
        self.epoch = 0
        self.nb

    def warm_up_step(self):
        self.step += 1

        xi = [0, self.warmup_steps]  # x interp
        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(self.step, xi, [
                self.warmup_bias_lr if j == 2 else 0.0,
                x['initial_lr'] * self.lf(self.epoch)
            ])
            if 'momentum' in x:
                x['momentum'] = np.interp(
                    self.step, xi, [self.warmup_momentum, self.momentum])

        if self.step % self.nb == 0:
            self.epoch += 1

    def step_epoch(self):
        self.one_cycle_lr_scheduler.step()

    def one_cycle(self, y1=0.0, y2=1.0, epochs=100):
        # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
        return lambda x: (
            (1 - math.cos(x * math.pi / epochs)) / 2) * (y2 - y1) + y1

    def one_cycle_lr_scheduler(self, hyp, epochs, optimizer):
        self.lf = self.one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)

        return scheduler


def one_cycle(y1=0.0, y2=1.0, epochs=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: (
        (1 - math.cos(x * math.pi / epochs)) / 2) * (y2 - y1) + y1