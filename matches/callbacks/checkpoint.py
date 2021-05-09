import logging

from ignite.distributed import one_rank_only

from .callback import Callback
from ..loop import Loop

LOG = logging.getLogger(__name__)


class BestModelSaver(Callback):
    def __init__(
        self, metric_name: str, metric_mode: str = "min", logdir_suffix: str = ""
    ):
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
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
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
        checkpoint_path = loop.logdir / self.logdir_suffix / "best.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        loop.state_manager.write_state(checkpoint_path)

    def load_best_model(self, loop: Loop):
        checkpoint_path = loop.logdir / self.logdir_suffix / "best.pth"

        loop.state_manager.read_state(checkpoint_path)


class LastModelSaverCallback(Callback):
    def __init__(self, logdir_suffix: str = ""):
        self.logdir_suffix = logdir_suffix

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self.save_model(loop)

    def save_model(self, loop: Loop):
        checkpoint_path = loop.logdir / self.logdir_suffix / "last.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        loop.state_manager.write_state(checkpoint_path)
