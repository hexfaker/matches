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
    r"""
    Collects metrics on batch and epochs
    """

    def __init__(self, loop: "Loop"):
        self._loop = loop
        self._new_entries: List[MetricEntry] = []
        self.latest: Dict[str, MetricEntry] = {}

    def reset(self):
        self._new_entries = []
        self.latest = {}

    def collect_new_entries(self, reset=True) -> List[MetricEntry]:
        result = self._new_entries
        if reset:
            self._new_entries = []
        return result


    def _guess_iteration_type(self):
        if self._loop.current_dataloader is not None:
            return MetricIterationType.BATCHES
        return MetricIterationType.EPOCHS

    def log(
        self,
        name: str,
        value: Union[float, torch.Tensor, np.ndarray],
        iteration: Union[str, MetricIterationType, int] = MetricIterationType.AUTO,
    ):
        r"""Logs metric value at specified iteration

        Adds or updates metric to :attr:`MetricManager.latest` with key :param:`name`
        and appends to updates list.

        Note:
              For `Metric` objects see :meth:`MetricManager.consume`.

        Note:
            :param:`iteration` parameter controls how metric values are distinguished in time.

            * By default metric iteration is chosen automatically: If it's called inside
              dataloader loop iteration assumed `"batches"`, otherwise it's assumed `"epochs"`.
            * Metrics like loss on current batch should set :param:`iteration` to
              `MetricIterationType.BATCHES` or `"batches"`.
            * Metrics like F1-score on validation dataset should set it to
              `MetricIterationType.EPOCHS` or `"epochs"`.
            * Also `MetricIterationType.SAMPLES` or `"samples"` is available and equals to
              number of processed training samples. It can be useful eg when you want to
              compare data efficiency of different models.
            * If you want to set some specific iteration number you can pass integer instead
              of enum or str. Metric manager will assume this is some custom iteration mode.
            * Anyway, metric values are stored with all possible iteration values and passes
              chosen by you to loggers supporting only one value (eg :obj:`TensorboardMetricWriter`)

        Args:
            name: Metric name
            value: Metric value. Can be scalar torch.Tensor, np.ndarray or float
            iteration: Iteration on which metric is recorded. See Note

        Returns:
              None
        """
        if isinstance(iteration, int):
            iteration_type = MetricIterationType.CUSTOM
        else:
            iteration_type = MetricIterationType(iteration)

        if iteration_type == MetricIterationType.AUTO:
            iteration_type = self._guess_iteration_type()

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
    # TODO Raise warn if value in latest has different iteration type

    def consume(
        self,
        name: str,
        metric: Metric,
        iteration: Union[str, MetricIterationType, int] = MetricIterationType.AUTO,
    ):
        r"""Shortcut for logging and resetting metric in one call

        Shortcut for common scenario. Computes metric value, logs it and then resets it to prepare
        for new updates.

        If you don't need reset call ordinary :meth:`log`::

            metrics.log(name, metric.compute(),...)

        Args:
            name: Metric name
            metric: Metric object to log and reset
            iteration: Iteration on which metric is recorded. See Note
        Returns:
              None
        """
        self.log(name, metric.compute(), iteration=iteration)
        metric.reset()
