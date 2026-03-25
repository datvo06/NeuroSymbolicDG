import argparse
import math
import time
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from neurosymbolic_da.data.loader_utils import get_n_classes
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.adapt import freeze_structure, get_adaptable_params
from neurosymbolic_da.training.adversarial import (
    DomainDiscriminator,
    GradientReversalLayer,
    cdan_condition,
)
from neurosymbolic_da.training.losses import im_loss, l2sp_loss
from neurosymbolic_da.training.trainer import evaluate
from scripts.adapt_cdan_dg import get_dg_adapt_loaders


def main():
    device = torch.device("cuda")
    n_classes = get_n_classes("terrainc")

    # Build model
    model = NeuroSymbolicPipeline(
        n_primitives=8,
        n_classes=n_classes,
        backbone_variant="resnet50",
        pretrained_backbone=False,
        use_sparsemax=True,
        max_depth=1,
    )
    checkpoint_root = "checkpoints"
    all_ckpts = os.listdir(checkpoint_root)
    for ckpt in all_ckpts:
        target_name = "location_" + ckpt.split('.')[0].split('_')[-1]
        print(f"Evaluating on target domain {target_name} of checkpoint {ckpt}")
        _, _, tgt_test = get_dg_adapt_loaders(
            "./data/terra_incognita", target_name,
            batch_size=256, num_workers=16,
            dataset="terrainc",
        )
        ckpt = torch.load(os.path.join(checkpoint_root, ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        _, acc = evaluate(model, tgt_test, device)
        print(f"Accuracy {acc}")
        print("="*60)
if __name__ == "__main__":
    main()