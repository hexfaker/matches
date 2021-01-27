from collections import defaultdict
from enum import Enum
from typing import Union


class IterationType(Enum):
    AUTO = "auto"
    EPOCHS = "epochs"
    BATCHES = "batches"
    SAMPLES = "samples"
    CUSTOM = "custom"

    GLOBAL_EPOCHS = "global_epochs"
    GLOBAL_BATCHES = "global_batches"
    GLOBAL_STEPS = "global_steps"
    GLOBAL_SAMPLES = "global_samples"


class IterationCounter:
    class _ManagedInt(int):
        key: Union[str, IterationType]
        _manager: "IterationCounter"

        # noinspection PyArgumentList
        def __new__(cls, value, key, manager):
            obj = int.__new__(cls, value)
            cls.__init__(obj, value, key, manager)
            return obj

        def __init__(self, value, key, manager):
            super().__init__()
            self._key = key
            self._manager = manager

        def inc(self, iters=1):
            self._manager._storage[self._key] += iters

        def reset(self):
            self._manager._storage[self._key] = 0

        def remove(self):
            del self._manager._storage[self._key]

    def __init__(self):
        self._storage = defaultdict(int)

    def __getitem__(self, item):
        v = self._storage[item]
        return self._ManagedInt(v, item, self)

    @property
    def current_epoch(self) -> _ManagedInt:
        return self[IterationType.EPOCHS]

    @property
    def current_batch(self) -> _ManagedInt:
        return self[IterationType.BATCHES]

    @property
    def current_samples(self) -> _ManagedInt:
        return self[IterationType.SAMPLES]

    @property
    def global_steps(self) -> _ManagedInt:
        return self[IterationType.GLOBAL_STEPS]
