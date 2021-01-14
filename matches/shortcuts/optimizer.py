import typing
from typing import Protocol

import torch
from torch.optim.optimizer import Optimizer


@typing.runtime_checkable
class LRSchedulerProto(Protocol):
    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        pass

    def step(self, epoch=None):
        pass


def simple_gd_step(optimizer: Optimizer, loss: torch.Tensor, **backward_kwargs):
    optimizer.zero_grad()
    loss.backward(**backward_kwargs)
    optimizer.step()
