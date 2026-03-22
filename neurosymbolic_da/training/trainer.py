"""Source training and evaluation for the neuro-symbolic pipeline.

Implements Phase 1 (Section 4.3): train the full pipeline end-to-end
with classification loss on labeled source data.

    L_source = -sum_{(x,y)} log( W_y(x) / sum_{y'} W_{y'}(x) )

This is standard cross-entropy (NLL loss on log-softmax output).
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.losses import bottleneck_reg_loss, bottleneck_reg_loss_v2


@dataclass
class TrainMetrics:
    """Metrics accumulated during training."""

    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    epoch_time: float = 0.0
    history: list[dict] = field(default_factory=list)

    def record(self):
        self.history.append({
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "epoch_time": self.epoch_time,
        })


def train_epoch(
    model: NeuroSymbolicPipeline,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    grammar_l1: float = 0.0,
    bottleneck_reg: float = 0.0,
    bottleneck_reg_version: int = 1,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        grammar_l1: L1 sparsity penalty on grammar log_weights (0 = disabled)
        bottleneck_reg: weight for bottleneck diversity+concentration loss (0 = disabled).
            Forces concept bottleneck to learn spatially diverse, peaked part detectors.
        label_smoothing: label smoothing factor (0 = disabled, 0.1 = typical)

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        if bottleneck_reg > 0 and hasattr(model, 'forward_with_heatmaps'):
            log_probs, heatmaps = model.forward_with_heatmaps(batch_x)
        else:
            log_probs = model(batch_x)
        if label_smoothing > 0:
            # cross_entropy with label_smoothing expects raw logits, but we have log_probs
            # Use KL-div formulation: smooth targets + nll
            n_classes = log_probs.size(-1)
            smooth_targets = torch.full_like(log_probs, label_smoothing / n_classes)
            smooth_targets.scatter_(1, batch_y.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / n_classes)
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        else:
            loss = nn.functional.nll_loss(log_probs, batch_y)
        if grammar_l1 > 0 and hasattr(model, 'grammar'):
            loss = loss + grammar_l1 * model.grammar.log_weights.abs().mean()
        if bottleneck_reg > 0 and hasattr(model, 'forward_with_heatmaps'):
            reg_fn = bottleneck_reg_loss_v2 if bottleneck_reg_version == 2 else bottleneck_reg_loss
            loss = loss + bottleneck_reg * reg_fn(heatmaps)
        # MoE bottleneck load balance loss
        if hasattr(model, 'bottleneck') and hasattr(model.bottleneck, 'load_balance_loss'):
            lb_loss = model.bottleneck.load_balance_loss
            if lb_loss.requires_grad or lb_loss.item() > 0:
                loss = loss + 0.01 * lb_loss
        # Orthogonal relation sparsity loss
        if hasattr(model, 'relation_params') and hasattr(model.relation_params, 'get_relation_sparsity_loss'):
            loss = loss + model.relation_params.get_relation_sparsity_loss()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = log_probs.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: NeuroSymbolicPipeline,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a dataset.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        log_probs = model(batch_x)
        loss = nn.functional.nll_loss(log_probs, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = log_probs.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def train(
    model: NeuroSymbolicPipeline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    n_epochs: int = 50,
    scheduler: LRScheduler | None = None,
    log_interval: int = 1,
    save_path: str | None = None,
    grammar_l1: float = 0.0,
    bottleneck_reg: float = 0.0,
    bottleneck_reg_version: int = 1,
    label_smoothing: float = 0.0,
) -> TrainMetrics:
    """Full source training loop (Phase 1).

    Args:
        model: the pipeline
        train_loader: source training data
        val_loader: source validation data (or target test data for monitoring)
        optimizer: optimizer
        device: torch device
        n_epochs: number of epochs
        scheduler: optional LR scheduler
        log_interval: print every N epochs
        save_path: if set, save best model checkpoint
        bottleneck_reg: weight for bottleneck diversity+concentration loss
        label_smoothing: label smoothing factor (0 = disabled)

    Returns:
        TrainMetrics with full history
    """
    model.to(device)
    metrics = TrainMetrics()
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device,
                                            grammar_l1=grammar_l1,
                                            bottleneck_reg=bottleneck_reg,
                                            bottleneck_reg_version=bottleneck_reg_version,
                                            label_smoothing=label_smoothing)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        metrics.epoch = epoch
        metrics.train_loss = train_loss
        metrics.train_acc = train_acc
        metrics.val_loss = val_loss
        metrics.val_acc = val_acc
        metrics.epoch_time = time.time() - t0
        metrics.record()

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"time={metrics.epoch_time:.1f}s"
            )

        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            for _attempt in range(3):
                try:
                    torch.save(ckpt, save_path)
                    break
                except RuntimeError:
                    import time as _t
                    _t.sleep(0.5)

    return metrics
