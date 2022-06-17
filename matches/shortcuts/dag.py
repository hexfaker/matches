import functools
import logging
from contextlib import contextmanager
from inspect import signature
from typing import Callable, TypeVar

import torch
from torch.autograd.profiler import record_function
from typing_extensions import ParamSpec

R = TypeVar("R")
Args = ParamSpec("Args")


def graph_node(
    method: Callable[Args, R],
) -> Callable[Args, R]:
    sig = signature(method)
    if "self" not in sig.parameters:
        raise ValueError(
            "graph_node can only be used on ComputationGraph subclass methods"
        )

    @functools.wraps(method)
    def _wrapper(self, **kwargs):
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

    def get_or_compute_node(self, method, **kwargs):
        args = tuple([(k, kwargs[k]) for k in sorted(kwargs.keys())])
        name = method.__name__
        key = (name, args)
        value = self.get_cache_entry(key)

        if value is not None:
            logging.debug("Node <%s> cache hit", key)
            if torch.is_tensor(value) and not torch.is_grad_enabled():
                # Ensure detached version is returned if gradient is disabled globally
                value = value.detach()
            return value

        logging.debug("Node <%s> compute", key)
        with record_function(name):
            value = method(self, **kwargs)
        self._cache[key] = value

        return value
