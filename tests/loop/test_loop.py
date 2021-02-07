import inspect
from typing import List

import pytest
from pytest import fixture
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from matches.callbacks import Callback
from matches.loop import Loop


class EventHistoryCallback(Callback):
    def __init__(self):
        self.history = []

    def _write_caller_event(self):
        self.history.append(inspect.stack()[1].function)

    def on_epoch_start(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._write_caller_event()

    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._write_caller_event()

    def on_dataloader_start(self, loop: "Loop", dataloader: DataLoader):
        self._write_caller_event()

    def on_dataloader_end(self, loop: "Loop", dataloader: DataLoader):
        self._write_caller_event()

    def on_iteration_start(self, loop: "Loop", batch_no: int):
        self._write_caller_event()

    def on_iteration_end(self, loop: "Loop", batch_no: int):
        self._write_caller_event()

    def on_train_start(self, loop: "Loop"):
        self._write_caller_event()

    def on_train_end(self, loop: "Loop"):
        self._write_caller_event()

    def on_after_backward(self, loop: "Loop"):
        self._write_caller_event()

    def on_before_optimizer_step(self, loop: "Loop", optimizer: Optimizer):
        self._write_caller_event()

    def on_after_optimizer_step(self, loop: "Loop", optimizer: Optimizer):
        self._write_caller_event()


def is_pair(start: str, end: str):
    return start[:-5] == end[:-3]


def assert_has_correct_event_history(history: List[str]):
    paired_events = [e for e in history if e.startswith("on")]

    stack = []
    for e in paired_events:
        if e.endswith("start"):
            stack.append(e)
        elif e.endswith("end"):
            assert is_pair(stack.pop(), e)
        else:
            assert False


@fixture
def history():
    return EventHistoryCallback()


@fixture
def loop(tmpdir, history):
    return Loop(tmpdir, [history])


FAKE_TRAIN_DL = [f"train_{i}" for i in range(3)]
FAKE_VALID_DL = [f"valid_{i}" for i in range(3)]


def test_normal_cycle(loop, history):
    def run(loop: Loop):
        for _ in loop.iterate_epochs(2):
            for batch in loop.iterate_dataloader(FAKE_TRAIN_DL, mode="train"):
                history.history.append(batch)

            for batch in loop.iterate_dataloader(FAKE_VALID_DL):
                history.history.append(batch)

    loop.run(run)

    assert_has_correct_event_history(history.history)


def test_with_continue(loop, history):
    def run(loop: Loop):
        for epoch in loop.iterate_epochs(3):
            history.history.append(f"epoch_{epoch}")
            for i, batch in enumerate(loop.iterate_dataloader(FAKE_TRAIN_DL, mode="train")):
                history.history.append(batch)
                if i == 1:
                    break

            for batch in loop.iterate_dataloader(FAKE_VALID_DL):
                history.history.append(batch)

            if epoch == 1:
                break

    loop.run(run)

    assert_has_correct_event_history(history.history)

    assert "train_1" in history.history
    assert "train_2" not in history.history
    assert "valid_2" in history.history
    assert history.history.count("valid_2") == 2
    assert "epoch_0" in history.history
    assert "epoch_1" in history.history
    assert "epoch_2" not in history.history


def test_exception_handled_correctly(loop, history):
    def run_exc_in_dataloader(loop: Loop):
        for epoch in loop.iterate_epochs(2):
            history.history.append(f"epoch_{epoch}")
            for i, batch in enumerate(loop.iterate_dataloader(FAKE_TRAIN_DL, mode="train")):
                history.history.append(batch)

                if i == 1:
                    raise ValueError()  # Just example exception to catch

    with pytest.raises(ValueError):
        loop.run(run_exc_in_dataloader)

    assert_has_correct_event_history(history.history)


def test_postphone_exception_in_callback():
    # TODO implement exception postphoning behaviour
    pass
