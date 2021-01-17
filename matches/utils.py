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
