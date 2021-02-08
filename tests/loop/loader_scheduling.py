import torch
from pytest import fixture
from torch.utils.data import DataLoader, TensorDataset

from matches.loop.loader_scheduling import DataloaderSchedulerWrapper


@fixture
def fake_loader():
    return DataLoader(TensorDataset(*torch.randn(10, 3, 10, 10)))

@fixture
def wrapper(fake_loader):
    return DataloaderSchedulerWrapper(fake_loader, single_pass_length=5, truncated_length=2)


def test_wrapper_seamless_attrs(wrapper):
    assert len(wrapper) == 5
    
    wrapper.num_workers = 3
    assert wrapper.dataloader.num_workers == 3

    wrapper.num_workers = 5
    assert wrapper.num_workers == 5
