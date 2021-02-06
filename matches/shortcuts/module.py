from contextlib import contextmanager

from torch import nn


@contextmanager
def no_grad_for_module(mod: nn.Module):
    """
    Context manager
    Sets requires_grad=False to all module's parameters
    on enter and restores value on exit
    Args:
        mod: module disable grads on

    Returns: None

    """
    old_state = {}
    try:
        for name, tensor in mod.named_parameters():
            old_state[name] = tensor.requires_grad
            tensor.requires_grad_(False)
        yield 
    finally:
        for name, tensor in mod.named_parameters():
            tensor.requires_grad_(old_state[name])


@contextmanager
def module_train(mod: nn.Module, train=True):
    """
    Context manager setting module and it's children
    train to specified value on enter,
    and restoring previous mode on exit
    Args:
        mod: module to work on
        train: mode to set

    Returns: None

    """
    old_state = {}
    try:
        for name, module in mod.named_modules():
            module: nn.Module = module
            old_state[name] = module.training
            module.train(train)
        yield
    finally:
        for name, module in mod.named_modules():
            module.train(old_state[name])

def module_eval(mod: nn.Module):
    return module_train(mod, False)
