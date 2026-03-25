import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DOMAINNET_INC_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


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


def _build_imgtxt_to_imgfolder_map(root: str, domain: str) -> dict[int, int]:
    """Build mapping from images.txt indices to ImageFolder indices.

    images.txt and ImageFolder have different sort orders within each class.
    split.json references images.txt ordering, so we need this mapping.
    """
    # Parse images.txt: 1-based ID -> relative path
    imgtxt_paths = {}
    with open(Path(root) / "images.txt") as f:
        for line in f:
            idx, path = line.strip().split(" ", 1)
            imgtxt_paths[int(idx)] = path  # e.g. "001.Black.../file.jpg"

    # Build ImageFolder and get its ordering
    from torchvision import datasets as _ds
    domain_path = Path(root) / domain
    ds = _ds.ImageFolder(str(domain_path), transform=None)
    # Map: relative filename -> ImageFolder index
    imgfolder_map = {}
    prefix = domain + "/"
    for i, (fpath, _) in enumerate(ds.imgs):
        rel = fpath.split(prefix, 1)[1]  # "001.Black.../file.jpg"
        imgfolder_map[rel] = i

    # Map images.txt 1-based ID -> ImageFolder 0-based index
    mapping = {}
    for txt_id, rel_path in imgtxt_paths.items():
        mapping[txt_id] = imgfolder_map[rel_path]
    return mapping


def _load_split_indices(root: str, domain: str, split: str = "train") -> list[int]:

    split_file = Path(root) / "split.json"
    with open(split_file) as f:
        splits = json.load(f)

    key = "train_valid" if split == "train" else "test"
    txt_indices = splits[key]  # 1-based images.txt indices

    # Map from images.txt ordering to ImageFolder ordering
    mapping = _build_imgtxt_to_imgfolder_map(root, domain)
    indices = [mapping[idx] for idx in txt_indices]
    return sorted(indices)


def get_domainnet(
    root: str,
    domain: str,
    train: bool = True,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> Subset:
    seed=42
    domain_path = Path(root) / domain    
    if not domain_path.exists():
        raise FileNotFoundError(
            f"DomainNet domain folder not found at {domain_path}. "
        )
    transform = _default_transform(
        image_size, 
        train=train, 
        strong_aug=strong_aug,
        randaugment=randaugment
    )
    
    full_dataset = datasets.ImageFolder(str(domain_path), transform=transform)
    num_items = len(full_dataset)
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    split_idx = int(num_items * 0.8) # 80% for training
    if train:
        target_indices = indices[:split_idx]
    else:
        target_indices = indices[split_idx:]
        
    return Subset(full_dataset, target_indices)


def get_domainnet_loaders(
    root: str,
    source: str,
    target: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get source/target train/test loaders for a domainnet transfer task.

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    src_train = get_domainnet(root, source, train=True, image_size=image_size,
                          strong_aug=strong_aug, randaugment=randaugment)
    src_test = get_domainnet(root, source, train=False, image_size=image_size)
    tgt_train = get_domainnet(root, target, train=True, image_size=image_size,
                          strong_aug=strong_aug, randaugment=randaugment)
    tgt_test = get_domainnet(root, target, train=False, image_size=image_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )
