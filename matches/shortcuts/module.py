from contextlib import contextmanager
from typing import Dict, List

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
def module_train(*modules: nn.Module, train=True):
    """
    Context manager setting module and it's children
    train to specified value on enter,
    and restoring previous mode on exit
    Args:
        modules: modules to work on
        train: mode to set

    Returns: None

    """
    old_states: List[Dict[str, bool]] = []

    try:
        for module in modules:
            old_state = {}
            for name, child in module.named_modules():
                old_state[name] = child.training
                child.training = train
            old_states.append(old_state)
        yield
    finally:
        for module, old_state in zip(modules, old_states):
            for name, child in module.named_modules():
                child.training = old_state[name]


def module_eval(mod: nn.Module):
    return module_train(mod, train=False)


