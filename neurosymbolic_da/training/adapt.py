"""Domain adaptation protocol (Phase 2, Section 4.3).

Freezes grammar weights and relation parameters, adapts only the
backbone and concept bottleneck using:

    L_adapt = L_MMD(h_s, h_t) + lambda * L_entropy(x_t)
"""

import time
from dataclasses import dataclass, field

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.losses import entropy_loss, im_loss, l2sp_loss, mmd_loss
from neurosymbolic_da.training.trainer import evaluate


@dataclass
class AdaptMetrics:
    """Metrics accumulated during adaptation."""

    epoch: int = 0
    mmd: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    target_acc: float = 0.0
    epoch_time: float = 0.0
    history: list[dict] = field(default_factory=list)

    def record(self):
        self.history.append({
            "epoch": self.epoch,
            "mmd": self.mmd,
            "entropy": self.entropy,
            "total_loss": self.total_loss,
            "target_acc": self.target_acc,
            "epoch_time": self.epoch_time,
        })


def freeze_structure(model: NeuroSymbolicPipeline, freeze_grammar: bool = True) -> None:
    """Freeze grammar weights and relation parameters.

    After this, only backbone and bottleneck parameters are trainable.

    Args:
        model: the pipeline
        freeze_grammar: if False, grammar weights remain trainable
            (ablation: "no grammar freeze" in Section 6.6)
    """
    if freeze_grammar:
        for param in model.grammar.parameters():
            param.requires_grad = False
    for param in model.relation_params.parameters():
        param.requires_grad = False


def get_adaptable_params(model: NeuroSymbolicPipeline) -> list[torch.nn.Parameter]:
    """Return only the parameters that should be adapted.

    Includes backbone + bottleneck always; also grammar if not frozen.
    """
    params = []
    for param in model.backbone.parameters():
        if param.requires_grad:
            params.append(param)
    for param in model.bottleneck.parameters():
        if param.requires_grad:
            params.append(param)
    for param in model.grammar.parameters():
        if param.requires_grad:
            params.append(param)
    return params


def adapt_epoch(
    model: NeuroSymbolicPipeline,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    lambda_entropy: float = 0.1,
    use_im_loss: bool = False,
    use_bottleneck_mmd: bool = False,
    source_params: dict[str, torch.Tensor] | None = None,
    lambda_l2sp: float = 0.0,
    use_production_mmd: bool = False,
    lambda_production_mmd: float = 0.1,
) -> tuple[float, float, float]:
    """Run one adaptation epoch.

    Args:
        model: the pipeline (with grammar/relation_params frozen)
        source_loader: labeled source data (only images used, labels ignored)
        target_loader: unlabeled target data
        optimizer: optimizer over adaptable params only
        device: torch device
        lambda_entropy: weight for entropy/IM loss
        use_im_loss: if True, use Information Maximization loss instead of
            plain entropy. IM adds a diversity term that prevents collapse.
        use_bottleneck_mmd: if True, align compact bottleneck features
            [B, k*3] instead of raw heatmaps [B, k*H*W].
        source_params: if set, dict of source parameter values for L2-SP
        lambda_l2sp: weight for L2-SP regularization
        use_production_mmd: if True, also align production scores via MMD
        lambda_production_mmd: weight for production score MMD

    Returns:
        (avg_mmd, avg_entropy, avg_total_loss)
    """
    model.train()
    total_mmd = 0.0
    total_ent = 0.0
    total_loss = 0.0
    n_batches = 0

    target_iter = iter(target_loader)

    for source_batch in source_loader:
        # Get source images (ignore labels)
        source_x = source_batch[0].to(device)

        # Get target images (cycle if target is shorter)
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        target_x = target_batch[0].to(device)

        optimizer.zero_grad()

        # Feature alignment via MMD
        if use_bottleneck_mmd:
            source_feats = model.get_bottleneck_features(source_x)
            target_feats = model.get_bottleneck_features(target_x)
            l_mmd = mmd_loss(source_feats, target_feats)
        else:
            source_heatmaps = model.get_heatmaps(source_x)
            target_heatmaps = model.get_heatmaps(target_x)
            l_mmd = mmd_loss(source_heatmaps, target_heatmaps)

        # Prediction loss on target
        target_log_probs = model(target_x)
        if use_im_loss:
            l_ent = im_loss(target_log_probs)
        else:
            l_ent = entropy_loss(target_log_probs)

        # Production score alignment: MMD on grammar production activations
        if use_production_mmd:
            source_prod = model.get_production_scores(source_x)
            target_prod = model.get_production_scores(target_x)
            l_prod_mmd = mmd_loss(source_prod, target_prod)
            loss = l_mmd + lambda_entropy * l_ent + lambda_production_mmd * l_prod_mmd
        else:
            loss = l_mmd + lambda_entropy * l_ent

        # L2-SP regularization: penalize deviation from source weights
        if source_params is not None and lambda_l2sp > 0:
            l_l2sp = l2sp_loss(model, source_params)
            loss = loss + lambda_l2sp * l_l2sp

        loss.backward()
        optimizer.step()

        total_mmd += l_mmd.item()
        total_ent += l_ent.item()
        total_loss += loss.item()
        n_batches += 1

    return total_mmd / n_batches, total_ent / n_batches, total_loss / n_batches


def adapt(
    model: NeuroSymbolicPipeline,
    source_loader: DataLoader,
    target_loader: DataLoader,
    target_test_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    n_epochs: int = 20,
    lambda_entropy: float = 0.1,
    scheduler: LRScheduler | None = None,
    log_interval: int = 1,
    save_path: str | None = None,
    use_im_loss: bool = False,
    use_bottleneck_mmd: bool = False,
    source_params: dict[str, torch.Tensor] | None = None,
    lambda_l2sp: float = 0.0,
    use_production_mmd: bool = False,
    lambda_production_mmd: float = 0.1,
) -> AdaptMetrics:
    """Full adaptation loop (Phase 2).

    Assumes grammar and relation params are already frozen
    (call freeze_structure() first).

    Args:
        model: the pipeline
        source_loader: source training data (images only, for MMD)
        target_loader: unlabeled target training data
        target_test_loader: target test data (for evaluation)
        optimizer: optimizer over adaptable params
        device: torch device
        n_epochs: number of adaptation epochs
        lambda_entropy: weight for entropy loss
        scheduler: optional LR scheduler
        log_interval: print every N epochs
        save_path: if set, save best model checkpoint

    Returns:
        AdaptMetrics with full history
    """
    model.to(device)
    metrics = AdaptMetrics()
    best_target_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        avg_mmd, avg_ent, avg_loss = adapt_epoch(
            model, source_loader, target_loader, optimizer, device, lambda_entropy,
            use_im_loss=use_im_loss, use_bottleneck_mmd=use_bottleneck_mmd,
            source_params=source_params, lambda_l2sp=lambda_l2sp,
            use_production_mmd=use_production_mmd,
            lambda_production_mmd=lambda_production_mmd,
        )

        # Evaluate on target test set
        _, target_acc = evaluate(model, target_test_loader, device)

        if scheduler is not None:
            scheduler.step()

        metrics.epoch = epoch
        metrics.mmd = avg_mmd
        metrics.entropy = avg_ent
        metrics.total_loss = avg_loss
        metrics.target_acc = target_acc
        metrics.epoch_time = time.time() - t0
        metrics.record()

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"MMD={avg_mmd:.4f} ent={avg_ent:.4f} loss={avg_loss:.4f} | "
                f"target acc={target_acc:.4f} | "
                f"time={metrics.epoch_time:.1f}s"
            )

        if save_path and target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_acc": target_acc,
            }, save_path)

    return metrics
