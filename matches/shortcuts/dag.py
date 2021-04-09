from contextlib import contextmanager
from inspect import signature
from types import FunctionType
from typing import Optional, Sequence

import torch


def graph_node(method: FunctionType, *, cache_key_args: Optional[Sequence[str]] = None):
    sig = signature(method)
    if "self" not in sig.parameters:
        raise ValueError("graph_node can only be used on ComputationGraph subclass methods")

    def _wrapper(self: ComputationGraph, **kwargs):
        return self.get_or_compute_node(method, **kwargs)

    return _wrapper


class ComputationGraph:
    def __init__(self):
        self._cache = None

    @contextmanager
    def cache_scope(self):
        if self._cache is not None:
            raise ValueError("Entering cache_scope() twice is not supported")

        try:
            self._cache = {}
            yield
        finally:
            self._cache = None

    def get_cache_entry(self, key):
        if self._cache is None:
            raise ValueError("Attempt to use cache outside cache_scope()")
        return self._cache.get(key, None)

    def get_or_compute_node(self, method: FunctionType, **kwargs):
        key = method.__name__
        value = self.get_cache_entry(key)

        if value is not None:
            if torch.is_tensor(value) and not torch.is_grad_enabled():
                # Ensure detached version is returned if gradient is disabled globally
                value = value.detach()
            return value

        value = method(self, **kwargs)
        self._cache[key] = value

        return value
