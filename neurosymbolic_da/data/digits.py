"""Digit datasets: MNIST, USPS, SVHN.

All available via torchvision. Images are resized to 224x224 (3-channel)
for compatibility with ResNet backbones, or 32x32 for LeNet.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_grayscale_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _make_svhn_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Default transforms (224x224 for ResNet)
_GRAYSCALE_TRANSFORM = _make_grayscale_transform(224)
_SVHN_TRANSFORM = _make_svhn_transform(224)


def get_mnist(root: str = "./data", train: bool = True, download: bool = True,
              image_size: int = 224):
    t = _GRAYSCALE_TRANSFORM if image_size == 224 else _make_grayscale_transform(image_size)
    return datasets.MNIST(root, train=train, download=download, transform=t)


def get_usps(root: str = "./data", train: bool = True, download: bool = True,
             image_size: int = 224):
    t = _GRAYSCALE_TRANSFORM if image_size == 224 else _make_grayscale_transform(image_size)
    return datasets.USPS(root, train=train, download=download, transform=t)


def get_svhn(root: str = "./data", split: str = "train", download: bool = True,
             image_size: int = 224):
    t = _SVHN_TRANSFORM if image_size == 224 else _make_svhn_transform(image_size)
    return datasets.SVHN(root, split=split, download=download, transform=t)


def get_digit_loaders(
    source: str,
    target: str,
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get source/target train/test loaders for a digit transfer task.

    Args:
        source: one of "mnist", "usps", "svhn"
        target: one of "mnist", "usps", "svhn"
        root: data directory
        batch_size: batch size
        num_workers: dataloader workers
        image_size: input image size (224 for ResNet, 32 for LeNet)

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    getters = {
        "mnist": lambda train: get_mnist(root, train=train, image_size=image_size),
        "usps": lambda train: get_usps(root, train=train, image_size=image_size),
        "svhn": lambda train: get_svhn(root, split="train" if train else "test",
                                       image_size=image_size),
    }

    src_train = getters[source](True)
    src_test = getters[source](False)
    tgt_train = getters[target](True)
    tgt_test = getters[target](False)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )
