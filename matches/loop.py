import logging
import typing
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, TYPE_CHECKING, TypeVar

import ignite.distributed as idist
import torch
from ignite.metrics import Metric
from ignite.utils import convert_tensor
from torch import nn
from torch.utils.data import DataLoader

from .accelerators import Accelerator
from .metric_manager import MetricManager

if TYPE_CHECKING:
    from .callbacks.callback import Callback

LOG = logging.getLogger(__name__)

T_batch = TypeVar("T_batch")


@typing.runtime_checkable
class StateSource(Protocol):
    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass


class StateManager:
    def __init__(self):
        self._state_sources: Dict[str, StateSource] = {}

    def attach(self, key: str, source: StateSource):
        self._state_sources[key] = source

    def state_dict(self):
        return {key: value.state_dict() for key, value in self._state_sources.items()}

    def load_state_dict(self, state_dict):
        for k, ss in self._state_sources.items():
            ss.load_state_dict(state_dict[k])


class Loop:
    def __init__(self, num_epochs: int, logdir: PathLike, callbacks: List["Callback"]):
        self.callbacks = callbacks
        self.num_epochs = num_epochs
        self.state_manager = StateManager()
        self.metrics = MetricManager(self)
        self.logdir: Path = Path(logdir)

        self.current_dataloader: Optional[DataLoader] = None

        self._in_epoch = False
        self.current_dataloader = False

        self._current_epoch: Optional[int] = None
        self._current_iteration: Optional[int] = None
        self._current_batch: Optional[int] = None
        self._current_sample: Optional[int]
        self._current_loader: Optional[DataLoader] = None

        self._modules: List[nn.Module] = []

    def _emit_event(self, event: str):
        for c in self.callbacks:
            getattr(c, event)(self)

    def _set_training(self, training):
        for m in self._modules:
            m.train(training)

    def attach(
        self,
        id: str = None,
        obj: Any = None,
        objects_dict: typing.Mapping[str, Any] = None,
        **kwargs
    ):
        objects_dict = objects_dict or {}
        if id and obj is not None:
            if isinstance(obj, nn.Module):
                self._modules.append(obj)
            if isinstance(obj, StateSource) and not isinstance(obj, Metric):
                self.state_manager.attach(id, obj)
            if isinstance(obj, Metric):
                self.metrics.register(id, obj)

        kwargs.update(objects_dict)
        for k, v in kwargs.items():
            self.attach(k, v)

    def iterate_epochs(self) -> Iterable[int]:
        for e in range(self.num_epochs):
            try:
                self.metrics.reset()
                self._in_epoch = True
                self._current_epoch = e
                self._emit_event("on_epoch_start")
                yield e
            finally:
                self.metrics.compute()
                self._emit_event("on_epoch_end")
                self._in_epoch = False
        self._current_epoch = None

    def iterate_dataloader(
        self, dataloader: DataLoader[T_batch], mode="valid", move_to_default_device=True
    ) -> Iterable[T_batch]:
        assert self._in_epoch

        self._mode = mode
        self.current_dataloader = dataloader
        self._emit_event("on_dataloader_start")

        self._current_iteration = 0
        if self._current_batch is None:
            self._current_batch = 0
        try:
            torch.set_grad_enabled(mode == "train")
            self._set_training(mode == "train")
            for batch in dataloader:
                try:
                    self._emit_event("on_iteration_start")
                    if move_to_default_device:
                        batch = convert_tensor(batch, idist.device())
                    yield batch
                finally:
                    self._emit_event("on_iteration_end")
                    
                    if self._mode == "train":
                        self._current_batch += 1
                    
                    self._current_iteration += 1
        finally:
            torch.set_grad_enabled(True)
            self._set_training(True)
            self._emit_event("on_dataloader_end")
            self.current_dataloader = None

    def run(self, training_procedure: typing.Callable):
        try:
            self._emit_event("on_train_start")
            training_procedure(self)
        except:
            LOG.exception("Stage raised exception")
        finally:
            self._emit_event("on_train_end")

    def launch(self, program: typing.Callable, accelerator: Accelerator):
        accelerator.execute(program, loop=self)
