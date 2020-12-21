import sys
from typing import Optional, TextIO

import tqdm.auto as tqdm

from .callback import Callback
from ..logging import configure_logging
from ..loop import Loop


class TqdmProgressCallback(Callback):
    def __init__(self, stream: TextIO = sys.stdout):
        self._stream = stream
        self.epoch_progress: Optional[tqdm.tqdm] = None
        self.loader_progress: Optional[tqdm.tqdm] = None
        configure_logging()

    def on_train_start(self, loop: Loop):
        self.epoch_progress = tqdm.tqdm(desc="Epochs", total=loop.num_epochs, file=self._stream)

    def on_train_end(self, loop: Loop):
        self.epoch_progress.close()
        self.epoch_progress = None

    def on_epoch_end(self, loop: Loop):
        self.epoch_progress.update(1)

    def on_dataloader_start(self, loop: Loop):
        self.loader_progress = tqdm.tqdm(
            desc=loop._mode,
            total=len(loop.current_dataloader),
            file=self._stream,
            leave=False,
        )

    def on_dataloader_end(self, loop: Loop):
        self.loader_progress.close()

    def on_iteration_end(self, loop: Loop):
        self.loader_progress.update(1)
