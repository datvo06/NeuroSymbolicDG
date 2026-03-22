"""Office-31 and Office-Home datasets.

Both use ImageFolder layout: domain/class_name/image.jpg.
Download scripts are in scripts/download_office31.sh and scripts/download_officehome.sh.
"""

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OFFICE31_DOMAINS = ("amazon", "dslr", "webcam")
OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real_World")


def _default_transform(
    image_size: int = 224, train: bool = True, strong_aug: bool = False,
    randaugment: bool = False,
) -> transforms.Compose:
    if train:
        aug_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        if randaugment:
            aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        if strong_aug:
            aug_list += [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if strong_aug:
            aug_list.append(transforms.RandomErasing(p=0.25))
        return transforms.Compose(aug_list)
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_office31(
    root: str,
    domain: str,
    train: bool = True,
    image_size: int = 224,
    val_split: float = 0.2,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> datasets.ImageFolder:
    """Load an Office-31 domain.

    Args:
        root: path to Office31/ directory
        domain: one of "amazon", "dslr", "webcam"
        train: if True return train split, else val split
        image_size: resize target
        val_split: fraction held out for validation

    Expected layout: root/amazon/images/backpack/*.jpg
    """
    domain_path = Path(root) / domain / "images"
    if not domain_path.exists():
        # Some versions have flat layout: root/amazon/backpack/*.jpg
        domain_path = Path(root) / domain
    if not domain_path.exists():
        raise FileNotFoundError(
            f"Office-31 domain not found at {domain_path}. "
            f"Run: bash scripts/download_office31.sh {root}"
        )

    transform = _default_transform(image_size, train=train, strong_aug=strong_aug,
                                    randaugment=randaugment)
    full_dataset = datasets.ImageFolder(str(domain_path), transform=transform)

    if val_split > 0:
        n_val = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
        return train_ds if train else val_ds
    return full_dataset


def get_officehome(
    root: str,
    domain: str,
    train: bool = True,
    image_size: int = 224,
    val_split: float = 0.2,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> datasets.ImageFolder:
    """Load an Office-Home domain.

    Args:
        root: path to OfficeHome/ directory
        domain: one of "Art", "Clipart", "Product", "Real_World"
        train: if True return train split, else val split
        image_size: resize target

    Expected layout: root/Art/Alarm_Clock/*.jpg
    """
    domain_path = Path(root) / domain
    if not domain_path.exists():
        raise FileNotFoundError(
            f"Office-Home domain not found at {domain_path}. "
            f"Run: bash scripts/download_officehome.sh {root}"
        )

    transform = _default_transform(image_size, train=train, strong_aug=strong_aug,
                                    randaugment=randaugment)
    full_dataset = datasets.ImageFolder(str(domain_path), transform=transform)

    if val_split > 0:
        n_val = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
        return train_ds if train else val_ds
    return full_dataset


def get_office_loaders(
    dataset: str,
    root: str,
    source: str,
    target: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get source/target train/test loaders for an Office transfer task.

    Args:
        dataset: "office31" or "officehome"
        root: path to dataset root directory
        source: source domain name
        target: target domain name
        batch_size: batch size
        num_workers: dataloader workers
        image_size: resize target

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    getter = get_office31 if dataset == "office31" else get_officehome

    src_train = getter(root, source, train=True, image_size=image_size, strong_aug=strong_aug,
                       randaugment=randaugment)
    src_test = getter(root, source, train=False, image_size=image_size)
    tgt_train = getter(root, target, train=True, image_size=image_size, strong_aug=strong_aug,
                       randaugment=randaugment)
    tgt_test = getter(root, target, train=False, image_size=image_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )
