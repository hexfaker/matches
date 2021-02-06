from pathlib import Path
from typing import Optional

import typer
from typer import Typer

from matches.accelerators import DDPAccelerator
from matches.callbacks import BestModelSaver, TqdmProgressCallback
from matches.callbacks.tensorboard import TensorboardMetricWriterCallback
from matches.loop import Loop
from matches.utils import unique_logdir

from .config import load_config
from .train import train


app = Typer()


@app.command()
def launch(
    config_path: Optional[Path] = typer.Option(None, "-C", "--config"),
    gpus: str = typer.Option(None, "-g"),
    dev: bool = False,
):
    config = load_config(config_path)
    loop = Loop(
        unique_logdir("logs/"),
        [
            TensorboardMetricWriterCallback(),
            BestModelSaver("generator/error_epoch"),
            TqdmProgressCallback(),
        ],
        dev,
    )

    loop.launch(train, DDPAccelerator(gpus), config=config)


if __name__ == "__main__":
    app()
