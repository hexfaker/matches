# Full-featured DCGAN example

Based on [pytorch example](https://github.com/pytorch/examples/tree/master/dcgan) 
and [ignite example](https://github.com/pytorch/ignite/tree/master/examples/gan)

## Features

* Unique timestamped logdir each run 
* Metrics and images are dumped to tensorboard
* DDP support
* Typed configuration
* Best model by generator error is saved

## Where to look

1. `dcgan/__main__.py` - entrypoint and argparser, `Loop` class and callbacks are 
    initialized here.
2. `dcgan/data.py` - dataset, dataloader and transforms initialization method
3. `dcgan/config.py` - typed config object containing all available parameters and 
    config file parsing procedure. Config example can be found at `configs/` dir.
4. `dcgan/model.py` - Generator and Discriminator model declarations.
5. `dcgan/train.py` - Training procedure. Contains training loop and all necessary 
    initialization.


## How to run
```shell
python -m dcgan -C <path-to-config> -g <gpu-ids>
python -m dcgan -C configs/mnist.py -g 0,1 # Train on mnist with DDP
python -m dcgan -g 2 # Train with default config (CIFAR-10) 
```
