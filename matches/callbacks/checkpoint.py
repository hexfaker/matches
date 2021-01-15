import logging
from os import PathLike
from pathlib import Path

import torch
from ignite.distributed import one_rank_only

from .callback import Callback
from ..loop import Loop

LOG = logging.getLogger(__name__)


class BestModelSaver(Callback):
    def __init__(self, metric_name: str, metric_mode: str = "min", logdir_suffix: str = ""):
        self.metric_name = metric_name
        self.logdir_suffix = logdir_suffix
        self.metric_mode = metric_mode

        if self.metric_mode == "min":
            self.best_value = float("+inf")
            self._sel = min
        else:
            self.best_value = float("-inf")
            self._sel = max

    @one_rank_only()
    def on_epoch_end(self, loop: Loop):
        current_value = loop.metrics.latest[self.metric_name].value

        better_value = self._sel(current_value, self.best_value)

        if better_value != self.best_value:
            self.best_value = better_value
            LOG.info(
                "Metric %s reached new best value %g, updating checkpoint",
                self.metric_name,
                self.best_value,
            )
            self.save_model(loop)

    def save_model(self, loop: Loop):

        state = loop.state_manager.state_dict()

        checkpoint_path = loop.logdir / self.logdir_suffix / "best.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with checkpoint_path.open("wb") as f:
            torch.save(state, f)
