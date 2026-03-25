import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VLCS_DOMAINS = ("CALTECH", "LABELME", "PASCAL", "SUN")


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
    """Load image indices from split.json, mapped to ImageFolder ordering.

    Args:
        root: path to vlcs directory containing split.json
        domain: domain name for ImageFolder mapping
        split: "train" uses train_valid, "test" uses test

    Returns:
        list of 0-based indices into the ImageFolder dataset
    """
    split_file = Path(root) / "split.json"
    with open(split_file) as f:
        splits = json.load(f)

    key = "train_valid" if split == "train" else "test"
    txt_indices = splits[key]  # 1-based images.txt indices

    # Map from images.txt ordering to ImageFolder ordering
    mapping = _build_imgtxt_to_imgfolder_map(root, domain)
    indices = [mapping[idx] for idx in txt_indices]
    return sorted(indices)


def get_vlcs(
    root: str,
    domain: str,
    train: bool = True,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> Subset:
    """Load a vlcs domain with official train/test split.

    Args:
        root: path to vlcs directory
        domain: one of ["CALTECH", "LABELME", "PASCAL", "SUN"]
        train: if True return train split, else test split
        image_size: resize target

    Expected layout: root/{Photo,Art,Cartoon,Paint}/001.Black_footed_Albatross/*.jpg
    """
    split_name = "train" if train else "test"
        
    domain_path = Path(root) / domain / split_name
    
    if not domain_path.exists():
        raise FileNotFoundError(
            f"VLCS domain split not found at {domain_path}. "
            f"Check if the domain and split folders exist."
        )

    transform = _default_transform(
        image_size, 
        train=train, 
        strong_aug=strong_aug,
        randaugment=randaugment
    )
    
    dataset = datasets.ImageFolder(str(domain_path), transform=transform)

    return dataset


def get_vlcs_loaders(
    root: str,
    source: str,
    target: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get source/target train/test loaders for a vlcs transfer task.

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    src_train = get_vlcs(root, source, train=True, image_size=image_size,
                          strong_aug=strong_aug, randaugment=randaugment)
    src_test = get_vlcs(root, source, train=False, image_size=image_size)
    tgt_train = get_vlcs(root, target, train=True, image_size=image_size,
                          strong_aug=strong_aug, randaugment=randaugment)
    tgt_test = get_vlcs(root, target, train=False, image_size=image_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(src_train, shuffle=True, **kwargs),
        DataLoader(src_test, shuffle=False, **kwargs),
        DataLoader(tgt_train, shuffle=True, **kwargs),
        DataLoader(tgt_test, shuffle=False, **kwargs),
    )
