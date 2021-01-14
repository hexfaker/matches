from abc import ABC, abstractmethod
from collections import Callable
from typing import List, Union

import os
import torch
from ignite.distributed import Parallel


class Accelerator(ABC):
    @abstractmethod
    def execute(self, program: Callable, *args, **kwargs):
        pass


class DDPAccelerator(Accelerator):
    def __init__(self, devices: Union[str, List[int]] = None):
        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        if isinstance(devices, str):
            devices = list(map(int, devices.split(",")))
        self.devices = devices

    @staticmethod
    def _worker_fn(local_rank, func, *args, **kwargs):
        func(*args, **kwargs)

    def execute(self, func: Callable, *args, **kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.devices))
        with Parallel(backend="nccl", nproc_per_node=len(self.devices)) as p:
            p.run(self._worker_fn, func, *args, **kwargs)
