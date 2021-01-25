import os
from datetime import datetime
from os import PathLike
from pathlib import Path
import random
from typing import Union

import numpy as np
import torch


def unique_logdir(root: Union[PathLike, str], comment: str = ""):
    """Get unique logdir under root based on comment and timestamp"""
    now = datetime.now().strftime("%y%m%d_%H%M")

    return Path(root) / comment / now


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def setup_cudnn_reproducibility(deterministic: bool = None, benchmark: bool = None) -> None:
    """
    Prepares CuDNN benchmark and sets CuDNN
    to be deterministic/non-deterministic mode
    See https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    
    Borrowed https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/torch.py#L256

    Args:
        deterministic: deterministic mode if running in CuDNN backend.
        benchmark: If ``True`` use CuDNN heuristics to figure out
            which algorithm will be most performant
            for your model architecture and input.
            Setting it to ``False`` may slow down your training.
    """
    if torch.cuda.is_available():
        if deterministic is None:
            deterministic = (
                os.environ.get("CUDNN_DETERMINISTIC", "True") == "True"
            )
        torch.backends.cudnn.deterministic = deterministic

        if benchmark is None:
            benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        torch.backends.cudnn.benchmark = benchmark
