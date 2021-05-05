from itertools import islice
from typing import Generic, Iterable, TypeVar, Union

from copy import deepcopy
from torch.utils.data import DataLoader

T_co = TypeVar("T_co", covariant=True)


class DataloaderSchedulerWrapper(Generic[T_co]):
    """
    Dataloader wrapper allowing to truncate dataloader and\or run validation
    only after multiple full or single partial dataloader pass

    For example, with for dataloader with length=10_000
    single_pass_length=0.1 dataloader will return first 1k batches, second 1k batches on second
    pass etc.
    Truncated length allows to use only beginning samples from dataloader
    """

    def __init__(
        self,
        dataloader: DataLoader[T_co],
        *,
        single_pass_length: Union[int, float] = 1.0,
        truncated_length: Union[int, float] = 1.0,
    ):
        """

        Args:
            dataloader: dataloader to wrap
            single_pass_length: Consume only single_pass_length batches in single pass
            truncated_length: Consume only first truncated_length batches
              (or len(dataloader) * truncated_length) if it's float
        """

        self.dataloader = dataloader
        if isinstance(truncated_length, float):
            truncated_length = int(truncated_length * len(dataloader))

        assert truncated_length <= len(
            dataloader
        ), "Truncated length must be <=1.0 if float, or <= len(dataloader) if int"

        single_pass_length = single_pass_length
        if isinstance(single_pass_length, float):
            single_pass_length = int(single_pass_length * truncated_length)

        self.truncated_len = truncated_length
        self.single_pass_len = single_pass_length
        self._internal_iterator = None

        self._internal_loader_full_passes = 0
        self._internal_iteration = 0
        self._init_done = True

    def _internal_loader_iter(self):
        for i, batch in islice(enumerate(self.dataloader), self.truncated_len):
            self._internal_iteration = i
            yield batch
        self._internal_loader_full_passes += 1

    def __iter__(self) -> Iterable[T_co]:
        for it in range(self.single_pass_len):
            batch = None
            while not batch:
                if not self._internal_iterator:
                    self._internal_iterator = iter(self._internal_loader_iter())
                try:
                    batch = next(self._internal_iterator)
                except StopIteration:
                    self._internal_iterator = None

            yield batch

    def __len__(self):
        return self.single_pass_len

    def __getattr__(self, item):

        return getattr(self.dataloader, item)

    def __setattr__(self, key, value):
        if "_init_done" in self.__dict__ and key not in self.__dict__:
            setattr(self.dataloader, key, value)
        else:
            object.__setattr__(self, key, value)


class CacheSingleBatchDL():
    def __init__(self, loader):
        self.loader = loader
        self.batch = next(iter(loader))

    def __iter__(self) -> Iterable[T_co]:
        yield deepcopy(self.batch)

    def __len__(self):
        return 1

    def __getattr__(self, item):
        return getattr(self.dataloader, item)

    def __setattr__(self, key, value):
        if "_init_done" in self.__dict__ and key not in self.__dict__:
            setattr(self.dataloader, key, value)
        else:
            object.__setattr__(self, key, value)


class DataloaderOverrider:
    def __init__(self, mode="disabled"):
        self.mode = mode
        self.cache = {}

    def __call__(self, loader, mode="valid"):
        cache_key = hash((loader, mode))

        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.mode == "short":
            # Every extra worker slows down start
            # So we want minimal test of dataloader mutliprocessing,
            # But we want it fast
            loader.num_workers = 1
            loader = DataloaderSchedulerWrapper(loader, truncated_length=3)

        if self.mode == "overfit-batch":
            if "raw_loader" in self.cache:
                loader.num_workers = 1
                loader = self.cache["raw_loader"]
            else:
                loader = CacheSingleBatchDL(loader)
                self.cache["raw_loader"] = loader

            if mode == "train":
                loader.num_workers = 1
                loader = DataloaderSchedulerWrapper(
                    loader, single_pass_length=10., truncated_length=1
                )
            if mode == "valid":
                loader = DataloaderSchedulerWrapper(loader, truncated_length=1)

        self.cache[cache_key] = loader

        return loader
