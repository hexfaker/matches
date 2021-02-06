import ignite.distributed as idist
import torch
from ignite.distributed import auto_model, one_rank_only
from ignite.metrics import Average
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torchvision.utils import make_grid

from matches.loop import Loop
from matches.shortcuts.loop import get_summary_writer
from matches.shortcuts.module import module_eval, no_grad_for_module
from matches.shortcuts.optimizer import simple_gd_step
from matches.utils import seed_everything, setup_cudnn_reproducibility
from .config import Config
from .data import get_dataloader
from .model import Discriminator, Generator


def train(loop: Loop, config: Config):
    seed_everything(22)
    setup_cudnn_reproducibility(True)

    dataloader, num_channels = get_dataloader(config)

    generator = auto_model(Generator(config.z_dim, config.g_filters, num_channels))
    discriminator = auto_model(Discriminator(num_channels, config.d_filters))

    bce = BCEWithLogitsLoss()

    opt_G = Adam(generator.parameters(), lr=config.lr * idist.get_world_size(), betas=(config.beta_1, 0.999))
    opt_D = Adam(discriminator.parameters(), lr=config.lr * idist.get_world_size(), betas=(config.beta_1, 0.999))

    device = idist.device()
    real_labels = torch.ones(config.batch_size, device=device)
    fake_labels = torch.zeros(config.batch_size, device=device)
    fixed_noise = torch.randn(16, config.z_dim, 1, 1, device=device)

    def dump_fake_images_to_tb():
        with loop.mode("valid"):
            fake = make_grid(generator(fixed_noise), normalize=True, range=(-1, 1)).cpu()

        if idist.get_rank() == 0:
            sw: SummaryWriter = get_summary_writer(loop)
            sw.add_image("fake_images", fake, global_step=loop.iterations.current_epoch)
            
    def get_noise():
        return torch.randn(config.batch_size, config.z_dim, 1, 1, device=device)

    error_D_avg = Average()
    error_G_avg = Average()

    loop.attach(
        generator=generator,
        discriminator=discriminator,
        d_opt=opt_D,
        g_opt=opt_G
    )

    def stage_1(loop: Loop):
        for _ in loop.iterate_epochs(config.epochs):

            for real, _ in loop.iterate_dataloader(dataloader, mode="train"):
                output = discriminator(real)
                error_D_real = bce(output, real_labels)
                loop.backward(error_D_real)

                fake = generator(get_noise())

                # train with fake
                output = discriminator(fake.detach())
                error_D_fake = bce(output, fake_labels)

                loop.backward(error_D_fake)
                loop.optimizer_step(opt_D)

                with torch.no_grad():
                    error_D = error_D_fake + error_D_real
                    error_D_avg.update(error_D)

                with no_grad_for_module(discriminator), module_eval(discriminator):
                    # We don't want to compute grads for
                    # discriminator parameters on
                    # error_G backward pass
                    output = discriminator(fake)

                error_G = bce(output, real_labels)

                simple_gd_step(loop, opt_G, error_G)

                error_G_avg.update(error_G.detach())

                loop.metrics.log("generator/error_batch", error_G.item())
                loop.metrics.log("discriminator/error_batch", error_D.item())

            loop.metrics.consume("generator/error_epoch", error_G_avg)
            loop.metrics.consume("discriminator/error_epoch", error_D_avg)

            dump_fake_images_to_tb()

    loop.run(stage_1)
