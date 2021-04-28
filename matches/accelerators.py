import os
from abc import ABC, abstractmethod
from typing import Callable, List, Union

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

    def _master_port(self):
        return str(hash(tuple(sorted(self.devices))) % 1024 + 1024)

    @staticmethod
    def _worker_fn(local_rank, func, *args, **kwargs):
        func(*args, **kwargs)

    def execute(self, func: Callable, *args, **kwargs):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.devices))
        if len(self.devices) > 1:
            with Parallel(
                backend="nccl",
                nproc_per_node=len(self.devices),
                master_port=self._master_port(),
            ) as p:
                p.run(self._worker_fn, func, *args, **kwargs)
        else:
            func(*args, **kwargs)
