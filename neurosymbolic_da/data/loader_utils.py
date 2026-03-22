"""Shared dataset loading utilities for CLI scripts.

Centralizes dataset loading logic so all scripts (train_source, train_hybrid,
adapt_target, train_nopcfg, multi_adapt) use the same code paths.
"""

from torch.utils.data import DataLoader


def get_n_classes(dataset: str) -> int:
    match dataset:
        case "digits":
            return 10
        case "office31":
            return 31
        case "officehome":
            return 65
        case "scb":
            return 8
        case "cubdg":
            return 200
        case "pacs":
            return 7
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


def get_loaders(
    dataset: str,
    source: str,
    target: str,
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    scb_n_classes: int = 8,
    scb_n_samples: int = 200,
    scb_n_parts: int = 4,
    scb_image_size: int = 224,
    strong_aug: bool = False,
    randaugment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Load train/test loaders for source and target domains.

    Args:
        dataset: one of "digits", "office31", "officehome", "scb"
        source: source domain name (for scb: ignored, always "source")
        target: target domain name (for scb: condition "A", "B", or "C")
        data_root: root directory for data
        batch_size: batch size
        num_workers: dataloader workers
        scb_*: SCB-specific parameters

    Returns:
        (source_train, source_test, target_train, target_test)
    """
    if dataset == "digits":
        from neurosymbolic_da.data.digits import get_digit_loaders

        return get_digit_loaders(
            source, target,
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
    elif dataset == "scb":
        from neurosymbolic_da.data.scb import get_scb_loaders

        return get_scb_loaders(
            n_classes=scb_n_classes,
            n_samples_per_class=scb_n_samples,
            n_parts=scb_n_parts,
            image_size=scb_image_size,
            condition=target,  # target = condition for SCB
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "cubdg":
        from neurosymbolic_da.data.cubdg import get_cubdg_loaders

        return get_cubdg_loaders(
            root=data_root,
            source=source,
            target=target,
            batch_size=batch_size,
            num_workers=num_workers,
            strong_aug=strong_aug,
        )
    elif dataset == "pacs":
        from neurosymbolic_da.data.pacs import get_pacs_loaders

        return get_pacs_loaders(
            root=data_root,
            source=source,
            target=target,
            batch_size=batch_size,
            num_workers=num_workers,
            strong_aug=strong_aug,
            randaugment=randaugment,
        )
    else:
        from neurosymbolic_da.data.office import get_office_loaders

        return get_office_loaders(
            dataset, data_root, source, target,
            batch_size=batch_size,
            num_workers=num_workers,
            strong_aug=strong_aug,
            randaugment=randaugment,
        )
