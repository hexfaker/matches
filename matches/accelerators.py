import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import torch.cuda
from ignite.distributed import Parallel


class Accelerator(ABC):
    @abstractmethod
    def execute(self, program: Callable, *args, **kwargs):
        pass


def get_device_count_smi():
    output = subprocess.check_output(["nvidia-smi", "-L"])
    return len(output.splitlines())


class DDPAccelerator(Accelerator):
    def __init__(self, devices: Union[str, List[int]] = None):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            if devices is not None:
                logging.info("CUDA_VISIBLE_DEVICES is set. Other settings will be ignored")
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
        if devices is None:
            devices = list(range(get_device_count_smi()))

        if isinstance(devices, str):
            devices = list(map(int, devices.split(",")))
        self.devices = devices

    def _master_port(self) -> int:
        return int(str(hash(tuple(sorted(self.devices))) % 1024 + 1024))

    @staticmethod
    def _worker_fn(local_rank, func, *args, **kwargs):
        func(*args, **kwargs)

    def execute(self, func: Callable, *args, **kwargs):
        assert (
            not torch.cuda.is_initialized()
        ), "Accelerator failed to configure device visibility, cuda is already initialized. Please fix"
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
