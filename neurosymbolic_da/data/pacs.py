"""PACS dataset (Photo, Art_painting, Cartoon, Sketch).

4 domains, 7 classes, ~10K images. Standard DG benchmark.
Uses ImageFolder layout: domain/class_name/image.jpg.

Download via: huggingface-cli download flwrlabs/pacs --repo-type dataset
Or manually from https://sketchx.eecs.qmul.ac.uk/downloads/
"""

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PACS_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
PACS_CLASSES = 7  # dog, elephant, giraffe, guitar, horse, house, person


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


def get_pacs(
    root: str,
    domain: str,
    train: bool = True,
    image_size: int = 224,
    val_split: float = 0.2,
    strong_aug: bool = False,
    randaugment: bool = False,
):
    """Load a PACS domain.

    Args:
        root: path to PACS/ directory
        domain: one of "photo", "art_painting", "cartoon", "sketch"
        train: if True return train split, else val split
        image_size: resize target
        val_split: fraction held out for validation

    Expected layout: root/photo/dog/*.jpg
    """
    domain_path = Path(root) / domain
    if not domain_path.exists():
        raise FileNotFoundError(
            f"PACS domain not found at {domain_path}. "
            f"Download PACS to {root}"
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


def get_pacs_loaders(
    root: str,
    source: str,
    target: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get source/target train/test loaders for a PACS transfer task.

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    src_train = get_pacs(root, source, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
    src_test = get_pacs(root, source, train=False, image_size=image_size)
    tgt_train = get_pacs(root, target, train=True, image_size=image_size,
                         strong_aug=strong_aug, randaugment=randaugment)
    tgt_test = get_pacs(root, target, train=False, image_size=image_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )
