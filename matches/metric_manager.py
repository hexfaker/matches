import logging
from enum import Enum
from typing import Dict, List, TYPE_CHECKING, Union

import numpy as np

import torch
from dataclasses import dataclass
from ignite.metrics import Metric

if TYPE_CHECKING:
    from matches.loop import Loop

LOG = logging.getLogger(__name__)


class MetricIterationType(Enum):
    AUTO = "auto"
    EPOCHS = "epochs"
    BATCHES = "batches"
    SAMPLES = "samples"
    CUSTOM = "custom"


@dataclass
class MetricEntry:
    name: str
    value: float
    iteration_type: MetricIterationType
    iteration_values: Dict[MetricIterationType, int]

    @property
    def iteration(self):
        return self.iteration_values[self.iteration_type]


class MetricManager:
    """
    Collects metrics on batch and epochs
    """

    def __init__(self, loop: "Loop"):
        self._loop = loop
        self._metrics: Dict[str, Metric] = {}
        self._new_entries: List[MetricEntry] = []
        self.latest: Dict[str, MetricEntry] = {}

    def register(self, name: str, metric: Metric):
        self._metrics[name] = metric

    def reset(self):
        for m in self._metrics.values():
            m.reset()
        self._new_entries = []
        self.latest = {}

    def collect_new_entries(self, reset=True) -> List[MetricEntry]:
        result = self._new_entries
        if reset:
            self._new_entries = []
        return result

    def compute(self):
        epoch_values = {key: m.compute() for key, m in self._metrics.items()}

        for key, m in self._metrics.items():
            self.log(key, epoch_values[key], MetricIterationType.EPOCHS)

    def _guess_iteration_type(self):
        if self._loop._current_loader is not None:
            return MetricIterationType.BATCHES
        return MetricIterationType.EPOCHS

    def log(
        self,
        name: str,
        value: Union[float, torch.Tensor, Metric],
        iteration: Union[str, MetricIterationType, int] = MetricIterationType.AUTO,
    ):
        if isinstance(iteration, int):
            iteration_type = MetricIterationType.CUSTOM
        else:
            iteration_type = MetricIterationType(iteration)

        if iteration_type == MetricIterationType.AUTO:
            iteration_type = self._guess_iteration_type()

        if isinstance(value, Metric):
            metric = value
            value = value.compute()
            metric.reset()

        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            value = value.item()

        iteration_values = {
            MetricIterationType.EPOCHS: self._loop._current_epoch,
            MetricIterationType.BATCHES: self._loop._current_batch,
        }

        if iteration_type == MetricIterationType.CUSTOM:
            iteration_values[MetricIterationType.CUSTOM] = iteration

        entry = MetricEntry(name, value, iteration_type, iteration_values)
        self._new_entries.append(entry)

        self.latest[name] = entry


    # TODO Raise warn when adding metric with same name on same iteration twice
