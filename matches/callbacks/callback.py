import typing

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
