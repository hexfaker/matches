from ignite.distributed import auto_dataloader
from torchvision import transforms, datasets as dset
from ignite import distributed as idist

from .config import Config


def get_dataset(config: Config):
    resize = transforms.Resize(64)
    crop = transforms.CenterCrop(64)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if config.dataset == "mnist":
        normalize = transforms.Normalize((0.5), (0.5))

    if config.dataset in {"imagenet", "folder", "lfw"}:
        config.dataset = dset.ImageFolder(root=config.data_root, transform=transforms.Compose(
            [resize, crop, to_tensor, normalize]))
        num_channels = 3

    elif config.dataset == "lsun":
        config.dataset = dset.LSUN(
            root=config.data_root, classes=["bedroom_train"],
            transform=transforms.Compose([resize, crop, to_tensor, normalize])
        )
        num_channels = 3

    elif config.dataset == "cifar10":
        config.dataset = dset.CIFAR10(
            root=config.data_root, download=True,
            transform=transforms.Compose([resize, to_tensor, normalize])
        )
        num_channels = 3

    elif config.dataset == "mnist":
        config.dataset = dset.MNIST(root=config.data_root, download=True,
                             transform=transforms.Compose([resize, to_tensor, normalize]))
        num_channels = 1

    elif config.dataset == "fake":
        config.dataset = dset.FakeData(size=256, image_size=(3, 64, 64), transform=to_tensor)
        num_channels = 3

    else:
        raise RuntimeError(f"Invalid config.dataset name: {config.dataset}")

    return config.dataset, num_channels

def get_dataloader(config: Config):
    dataset, num_channels = get_dataset(config)

    loader = auto_dataloader(
        dataset,
        batch_size=config.batch_size * idist.get_world_size(),
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True
    )

    return loader, num_channels
