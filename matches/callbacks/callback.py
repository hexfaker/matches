import typing

from torch.optim import Optimizer

if typing.TYPE_CHECKING:
    from ..loop import Loop


class Callback:
    def on_epoch_start(self, loop: "Loop"):
        pass

    def on_epoch_end(self, loop: "Loop"):
        pass

    def on_dataloader_start(self, loop: "Loop"):
        pass

    def on_dataloader_end(self, loop: "Loop"):
        pass

    def on_iteration_start(self, loop: "Loop"):
        pass

    def on_iteration_end(self, loop: "Loop"):
        pass

    def on_train_start(self, loop: "Loop"):
        pass

    def on_train_end(self, loop: "Loop"):
        pass

    def on_after_backward(self, loop: "Loop"):
        pass

    def on_before_optimizer_step(self, loop: "Loop", optimizer: Optimizer):
        pass

    def on_after_optimizer_step(self, loop: "Loop", optimizer: Optimizer):
        pass
