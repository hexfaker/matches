import sys
from typing import Optional

import tqdm.auto as tqdm
from ignite.distributed import one_rank_only

from .callback import Callback
from ..logging import configure_logging
from ..loop import Loop


class TqdmProgressCallback(Callback):
    def __init__(self):
        pass

    @one_rank_only()
    def on_train_start(self, loop: Loop):
        # Save real strerr stream because it will be overridden in configure_logging()
        self._stream = sys.stdout
        self.epoch_progress: Optional[tqdm.tqdm] = tqdm.tqdm(
            desc="Epochs", file=self._stream
        )
        self.loader_progress: Optional[tqdm.tqdm] = tqdm.tqdm(
            file=self._stream,
            leave=False,
        )
        # Configure log sink, stdout and stderr to write with tqdm.write()
        # to keep things nice
        configure_logging(sys.stdout)
        
        self.epoch_progress.disable = False
        self.epoch_progress.reset(total=loop.num_epochs)

    @one_rank_only()
    def on_train_end(self, loop: Loop):
        self.epoch_progress = None

    @one_rank_only()
    def on_epoch_end(self, loop: Loop):
        self.epoch_progress.update(1)

    @one_rank_only()
    def on_dataloader_start(self, loop: Loop):
        self.loader_progress.reset(len(loop.current_dataloader))
        self.loader_progress.set_description(loop._mode)

    @one_rank_only()
    def on_iteration_end(self, loop: Loop):
        self.loader_progress.update(1)
