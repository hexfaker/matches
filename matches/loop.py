import logging
import typing
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
)

import ignite.distributed as idist
import torch
from ignite.metrics import Metric
from ignite.utils import convert_tensor
from torch import nn
from torch.optim import Optimizer
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
        """Object for metrics logging"""

        self.logdir: Path = Path(logdir)

        self.current_dataloader: Optional[DataLoader] = None

        self._in_epoch = False

        self._current_epoch: Optional[int] = None
        self._current_iteration: Optional[int] = None
        self._current_batch: Optional[int] = None
        self._current_sample: Optional[int]

        self._modules: List[nn.Module] = []

    def _emit_event(self, event: str, **event_kwargs):
        for c in self.callbacks:
            getattr(c, event)(self, **event_kwargs)

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
        """Add object to list of managed objects.

        Object with `state_dict()/load_state_dict()` will have their state saved and restored.
        For `torch.nn.Module` objects their `train()/eval()` mode will be managed.

        Examples::

            loop.attach("model", model)
            loop.attach(optimizer=optimizer)
            loop.attach(objects_dict={"model/generator": generator}

        Args:
            id: unique identifier for object
            obj: Object itself
            objects_dict: dict where keys are ids and values are objects
            **kwargs: arg-name -- id, arg-value -- object

        Returns:
              None

        """
        objects_dict = objects_dict or {}
        if id and obj is not None:
            if isinstance(obj, nn.Module):
                self._modules.append(obj)
            if isinstance(obj, StateSource) and not isinstance(obj, Metric):
                self.state_manager.attach(id, obj)

        kwargs.update(objects_dict)
        for k, v in kwargs.items():
            self.attach(k, v)

    def iterate_epochs(self) -> Iterable[int]:
        """Iterate over epochs

        * Emits callback events `on_epoch_start/end`

        Yields:
              current epoch number
        """
        for e in range(self.num_epochs):
            try:
                self.metrics.reset()
                self._in_epoch = True
                self._current_epoch = e
                self._emit_event("on_epoch_start")
                yield e
            finally:
                self._emit_event("on_epoch_end")
                self._in_epoch = False
        self._current_epoch = None

    def iterate_dataloader(
        self, dataloader: DataLoader[T_batch], mode="valid", move_to_default_device=True
    ) -> Iterable[T_batch]:
        """Iterate dataloader batches.

        Depending on chosen mode following is true inside loop over this generator:
        
        * Emits callback events `on_iteration_start/end`
        * Moves all torch.Tensor objects in batch to default device
        * Handles set_grad_enabled an train/eval of attached :obj:`nn.Module`:
        
            * mode="train" trigger train() on all `torch.nn.Modules` attached to
              `Loop`. Gradients are also enabled.
            * mode="valid" trigger eval() on all `torch.nn.Modules` attached to
              `Loop`. Gradients are also disabled.


        Args:
            dataloader: dataloader to iterate
            mode: "train" or "valid"
            move_to_default_device: Controls whether all tensor in batches are automatically moved
                to current device. Default `True`.

        Yields:
            Batches from dataloader.
        """
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

                    if self._mode == "train":
                        self._current_batch += 1
                finally:
                    self._emit_event("on_iteration_end")

                    self._current_iteration += 1
        finally:
            torch.set_grad_enabled(True)
            self._set_training(True)
            self._emit_event("on_dataloader_end")
            self.current_dataloader = None

    def backward(self, loss: torch.Tensor, **backward_kwargs):
        """Simple optional wrapper for backward call.

        Calls backward and emits event. Most likely will have more logic down the road.
        So it's recommended to use it even now

        Args:
            loss: loss to call backward on
            **backward_kwargs:

        Returns:
              None
        """
        loss.backward(**backward_kwargs)
        self._emit_event("on_after_backward")

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Optional[Callable[[], float]] = None,
        zero_grad: bool = True,
    ):
        """Optional shortcut wrapper for optimizer step.

        * Zeroes grad after step (quite common case, you know ;-))
        * Emits some callback events (before/after step)
        * Counts how much steps are done (differs from iterations/batches
          if you implement eg grad accumulation). This counter can be used as
          MetricsIterationType

        Args:
            optimizer: Optimizer to perform step
            closure: Some optimizers require closure
            zero_grad: Set False if you don't want to zero grad for some reason. True by default

        Returns:
              None
        """
        self._emit_event("on_before_optimizer_step", optimizer=optimizer)
        optimizer.step(closure)
        if zero_grad:
            optimizer.zero_grad()
        self._emit_event("on_before_optimizer_step", optimizer=optimizer)

    def run(self, training_procedure: typing.Callable):
        try:
            self._emit_event("on_train_start")
            training_procedure(self)
        except:
            LOG.exception("Stage raised exception")
        finally:
            self._emit_event("on_train_end")

    def launch(self, program: typing.Callable, accelerator: Accelerator):
        """Launch training program on chosen accelerator

        Examples::

            def train(loop: Loop):
                ...

            loop.launch(train, DDPAccelerator("gpu-id_1,gpu-id_2")

        Args:
              program: Training program
              accelerator: object defining accelerator environment


        """
        accelerator.execute(program, loop=self)
