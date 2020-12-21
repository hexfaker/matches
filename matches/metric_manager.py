import logging
from typing import Dict

import torch
from ignite.metrics import Metric

LOG = logging.getLogger(__name__)

class MetricManager:
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self.epoch_values: Dict[str, torch.Tensor] = {}

    def register(self, name: str, metric: Metric):
        self._metrics[name] = metric

    def reset(self):
        for m in self._metrics.values():
            m.reset()

    def compute(self):
        self.epoch_values = {key: m.compute() for key, m in self._metrics.items()}
