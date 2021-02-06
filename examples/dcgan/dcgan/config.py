from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Config(BaseModel):
    dataset: str = "cifar10"
    data_root: str = "data/"
    num_workers: int = 10
    batch_size: int = 64

    z_dim: int = 100
    g_filters: int = 64
    d_filters: int = 65
    epochs: int = 25
    beta_1: float = 0.5
    lr: float = 2e-4


def load_config(path: Optional[Path] = None):
    if path is None:
        return Config()

    text = path.read_text()

    ctx = {}
    exec(text, ctx)

    config = ctx["config"]

    assert config.__class__ is Config, (config.__class__, Config)
    return config

