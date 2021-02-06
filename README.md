# Matches. Next-generation pytorch train-loop
Take control back!

## Philosophy & Design

* Minimum inversion of control. Your code calls framework code, not vice versa.
  This approach provides almost same flexibility as pure PyTorch, but allows solving
  common problems like metrics, checkpoints, progress.

* Callbacks **only** for misc functionality (checkpoints, tensorboard, 
  benchmarking, debug etc).
  
* No oversimplification of most common cases like supervised learning.
  Other libraries out there do this, and as a result they have complex messy 
  internals and low flexibility for not-so-common usecase, like multiple optimizers 
  or multiple backward passes for single batch.

## Features
* Flexible metrics saving to tensorboard (batch/epoch/whatever)
* DDP
* Best model checkpoint by metric
* Automatic logdir naming
* Callback for checking that repo has no uncommitted changes
* Development mode to check pipeline fast
* Nice TQDM progress-bar that correctly handles printing  
* Simple and clear API. Also, it's quite future-proof and will have minimum 
  breakage over future development
* Comprehensive internals. Library code is clean and easy to understand, which
  prevents bugs and simplifies debugging if bug still happened
  
#### Planned
* DeepSpeed support
* Logging abstraction to seamlessly work with multiple sinks (eg tensorboard and WandB)
* Training stages/phases
* Rich configuration API with CLI support
* Train resuming

## Installation
```shell
pip install git+https://github.com/hexfaker/matches
```

## Examples
Examples are located in `examples/` dir. The most featured and sophisticated in 
[DCGAN example](https://github.com/hexfaker/matches/tree/master/examples/dcgan).
Others are simpler and less featured but can be still interesting.
