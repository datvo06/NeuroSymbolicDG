"""Test-time augmentation (TTA) and test-time adaptation (TENT/MEMO).

Implements three test-time strategies for improving DG generalization:

1. TTA: Average predictions over N augmented views of each image
2. TENT: Update BatchNorm affine parameters to minimize prediction entropy
   on test batches (Wang et al., ICLR 2021)
3. MEMO: Single-sample adaptation via marginal entropy minimization over
   augmented views (Zhang et al., NeurIPS 2022)
"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# TTA augmentation transforms
# ---------------------------------------------------------------------------

def get_tta_transforms(image_size: int = 224) -> list[Callable]:
    """Return a list of TTA transforms. Each takes a normalized tensor [3,H,W]
    and returns a normalized tensor [3,H,W].

    We work on already-normalized tensors to avoid re-loading images.
    Transforms that need PIL are avoided; we use tensor-level ops.
    """
    # Inverse normalization to get back to [0,1] tensor
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1.0 / s for s in IMAGENET_STD]
    inv_normalize = transforms.Normalize(inv_mean, inv_std)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def identity(x: Tensor) -> Tensor:
        return x

    def hflip(x: Tensor) -> Tensor:
        return x.flip(-1)  # flip width dimension

    def make_five_crop(idx: int):
        """Five-crop: TL, TR, BL, BR, center."""
        crop_size = image_size
        def crop(x: Tensor) -> Tensor:
            _, h, w = x.shape
            positions = [
                (0, 0),                         # TL
                (0, w - crop_size),              # TR
                (h - crop_size, 0),              # BL
                (h - crop_size, w - crop_size),  # BR
                ((h - crop_size) // 2, (w - crop_size) // 2),  # center
            ]
            top, left = positions[idx]
            return x[:, top:top + crop_size, left:left + crop_size]
        return crop

    def make_scale_crop(scale: float):
        """Resize to scale * 256, then center crop to image_size."""
        new_size = int(256 * scale)
        def transform(x: Tensor) -> Tensor:
            # x is [3, H, W] normalized — work in this space
            x_resized = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(new_size, new_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            # Center crop
            start = (new_size - image_size) // 2
            return x_resized[:, start:start + image_size, start:start + image_size]
        return transform

    # We need input at 256x256 for five-crop to work
    # Standard eval already resizes to 256x256 and center-crops to 224
    # For TTA, we take the 256x256 version and apply different crops
    # But since we get already-cropped 224x224 tensors from the dataloader,
    # we use scale + flip transforms instead

    tta_transforms = [
        identity,                    # original
        hflip,                       # horizontal flip
        make_scale_crop(0.875),      # smaller crop
        make_scale_crop(1.125),      # larger crop (will pad with interpolation)
        lambda x: hflip(make_scale_crop(0.875)(x)),  # flip + small
        lambda x: hflip(make_scale_crop(1.125)(x)),  # flip + large
    ]
    return tta_transforms


# ---------------------------------------------------------------------------
# TTA evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_views: int = 6,
) -> tuple[float, float]:
    """Evaluate with test-time augmentation.

    Averages log-softmax predictions over N augmented views.

    Args:
        model: the model (stays in eval mode)
        loader: test data loader
        device: torch device
        n_views: number of augmented views (max 6)

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    tta_transforms = get_tta_transforms()[:n_views]

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        B = batch_x.size(0)

        # Accumulate log-probs across views
        all_log_probs = []
        for tfm in tta_transforms:
            # Apply transform to each image in batch
            augmented = torch.stack([tfm(img) for img in batch_x])
            log_probs = model(augmented)
            all_log_probs.append(log_probs)

        # Average in probability space, then back to log
        avg_probs = torch.stack([lp.exp() for lp in all_log_probs]).mean(dim=0)
        avg_log_probs = (avg_probs + 1e-8).log()

        loss = nn.functional.nll_loss(avg_log_probs, batch_y)
        total_loss += loss.item() * B
        preds = avg_log_probs.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += B

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# TENT: Test-time Entropy minimization (Wang et al., ICLR 2021)
# ---------------------------------------------------------------------------

def collect_bn_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all BatchNorm affine parameters (weight and bias)."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def configure_tent(model: nn.Module) -> nn.Module:
    """Configure model for TENT: enable BN stats update, freeze all except BN affine."""
    model.eval()  # start from eval mode
    # Freeze everything
    model.requires_grad_(False)
    # Enable BN layers to update running stats and allow gradient on affine params
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()  # update running stats
            m.requires_grad_(True)
    return model


def tent_adapt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    n_steps: int = 1,
) -> tuple[float, float]:
    """Adapt model with TENT and evaluate.

    Updates BN affine params to minimize entropy on test batches.
    Each batch is processed for n_steps of gradient descent.

    Args:
        model: the model (will be modified in-place)
        loader: test data loader
        device: torch device
        lr: learning rate for BN param updates
        n_steps: gradient steps per batch

    Returns:
        (avg_loss, accuracy)
    """
    model = configure_tent(model)
    bn_params = collect_bn_params(model)
    optimizer = torch.optim.Adam(bn_params, lr=lr)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        B = batch_x.size(0)

        # Adapt BN params on this batch
        for _ in range(n_steps):
            log_probs = model(batch_x)
            # Entropy minimization
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

        # Evaluate after adaptation
        with torch.no_grad():
            log_probs = model(batch_x)
            loss = nn.functional.nll_loss(log_probs, batch_y)
            total_loss += loss.item() * B
            preds = log_probs.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += B

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# MEMO: Marginal Entropy Minimization with One test point
# (Zhang et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

def memo_adapt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 5e-3,
    n_steps: int = 1,
    n_augmentations: int = 6,
) -> tuple[float, float]:
    """Adapt model with MEMO and evaluate.

    For each test sample, generates augmented views and minimizes
    the marginal entropy (entropy of the averaged prediction).
    Only BN affine params are updated.

    Args:
        model: the model (BN params will be adapted per-sample)
        loader: test data loader (should have batch_size=1 ideally, but works with any)
        device: torch device
        lr: learning rate
        n_steps: gradient steps per sample
        n_augmentations: number of augmented views

    Returns:
        (avg_loss, accuracy)
    """
    tta_transforms = get_tta_transforms()[:n_augmentations]

    # Save original BN state to reset between samples
    original_state = copy.deepcopy(model.state_dict())

    total_loss = 0.0
    correct = 0
    total = 0

    model = configure_tent(model)
    bn_params = collect_bn_params(model)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        B = batch_x.size(0)

        # Process each sample individually for MEMO
        batch_preds = []
        batch_log_probs = []

        for i in range(B):
            # Reset BN params to original for each sample
            model.load_state_dict(original_state, strict=True)
            model = configure_tent(model)
            bn_params = collect_bn_params(model)
            optimizer = torch.optim.Adam(bn_params, lr=lr)

            img = batch_x[i:i + 1]  # [1, 3, H, W]

            # Adapt
            for _ in range(n_steps):
                # Generate augmented views
                aug_imgs = torch.cat([tfm(img[0]).unsqueeze(0) for tfm in tta_transforms])
                log_probs = model(aug_imgs)  # [n_aug, n_classes]
                # Marginal entropy: entropy of averaged prediction
                avg_probs = log_probs.exp().mean(dim=0)  # [n_classes]
                marginal_entropy = -(avg_probs * (avg_probs + 1e-8).log()).sum()
                optimizer.zero_grad()
                marginal_entropy.backward()
                optimizer.step()

            # Evaluate after adaptation
            with torch.no_grad():
                log_probs = model(img)  # [1, n_classes]
                batch_log_probs.append(log_probs)
                batch_preds.append(log_probs.argmax(dim=-1))

        log_probs_all = torch.cat(batch_log_probs)  # [B, n_classes]
        preds_all = torch.cat(batch_preds)  # [B]
        loss = nn.functional.nll_loss(log_probs_all, batch_y)
        total_loss += loss.item() * B
        correct += (preds_all == batch_y).sum().item()
        total += B

    # Restore original state
    model.load_state_dict(original_state, strict=True)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# TENT + TTA combo
# ---------------------------------------------------------------------------

def tent_tta_adapt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    n_steps: int = 1,
    n_views: int = 6,
) -> tuple[float, float]:
    """TENT adaptation with TTA at evaluation.

    First adapts BN via entropy minimization, then evaluates with TTA.
    Best of both worlds.

    Returns:
        (avg_loss, accuracy)
    """
    model = configure_tent(model)
    bn_params = collect_bn_params(model)
    optimizer = torch.optim.Adam(bn_params, lr=lr)
    tta_transforms = get_tta_transforms()[:n_views]

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        B = batch_x.size(0)

        # Step 1: TENT — adapt BN on this batch
        for _ in range(n_steps):
            log_probs = model(batch_x)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

        # Step 2: TTA — evaluate with augmented views
        with torch.no_grad():
            all_log_probs = []
            for tfm in tta_transforms:
                augmented = torch.stack([tfm(img) for img in batch_x])
                log_probs = model(augmented)
                all_log_probs.append(log_probs)
            avg_probs = torch.stack([lp.exp() for lp in all_log_probs]).mean(dim=0)
            avg_log_probs = (avg_probs + 1e-8).log()

        loss = nn.functional.nll_loss(avg_log_probs, batch_y)
        total_loss += loss.item() * B
        preds = avg_log_probs.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += B

    return total_loss / total, correct / total
