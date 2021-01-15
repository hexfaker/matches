from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from . import Callback
from ..loop import Loop


class TensorboardMetricWriterCallback(Callback):
    def __init__(self, logdir_suffix: str = ""):
        self.logdir_suffix = logdir_suffix

        self.sw: Optional[SummaryWriter] = None

    def on_iteration_end(self, loop: "Loop"):
        self._consume_new_entries(loop)

    def on_epoch_end(self, loop: "Loop"):
        self._consume_new_entries(loop)
        self.sw.flush()

    def _init_sw(self, loop: Loop):
        if self.sw is None:
            path = loop.logdir / self.logdir_suffix
            path.mkdir(exist_ok=True, parents=True)
            self.sw = SummaryWriter(str(path))

    def _consume_new_entries(self, loop):
        self._init_sw(loop)
        entries = loop.metrics.collect_new_entries()
        for e in entries:
            self.sw.add_scalar(e.name, e.value, global_step=e.iteration)
