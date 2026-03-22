# Neuro-Symbolic Domain Generalization via Compositional Layout Grammars

Abstract: A neuro-symbolic framework for domain generalization that factors visual recognition into **domain-invariant structural programs** (how parts compose into wholes via a PCFG grammar) and **domain-specific primitive detectors** (what parts look like). The grammar's compositional spatial reasoning is inherently domain-invariant, enabling strong generalization without explicit alignment losses.


## Setup

Requires Python >= 3.12.

```bash
# Install with uv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

Dependencies: `effectful`, `torch`, `torchvision`, `kornia`, `gdown`.

## Data
We provide scripts for downloading all data at once: 
```bash
bash data/get_all_data.sh
```

Or download each dataset individually:
```bash
# For example download the CUB-DG dataset:
bash data/cubdg.sh
```

## Reproducing Main Results

### 1. DG-ERM Source Training (train on 3 domains, test on held-out target)

```bash
# Best recipe: RandAugment + Label Smoothing + Sparsemax
for TGT in Art Cartoon Paint; do
    python scripts/train_dg.py \
        --dataset cubdg --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --pretrained \
        --n-primitives 8 --max-depth 1 --use-sparsemax --grammar-l1 0.01 \
        --randaugment --label-smoothing 0.1 \
        --epochs 50 --batch-size 32 --lr 1e-3 --num-workers 4 \
        --save-path checkpoints/dg_erm_v2_pcfg_cubdg_${TGT}.pt
done
```

Expected: Art 59.9%, Cartoon 66.5%, Paint 44.3% (avg 56.9%)

### 2. CDAN Adaptation (adapt to target with unlabeled target data)

```bash
# PCFG+CDAN -- BEST OVERALL: 67.0% avg
for TGT in Art Cartoon Paint; do
    python scripts/adapt_cdan_dg.py \
        --checkpoint checkpoints/dg_erm_v2_pcfg_cubdg_${TGT}.pt \
        --dataset cubdg --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --use-sparsemax --align-level backbone \
        --epochs 20 --batch-size 32 --lr 1e-4 --lr-disc 1e-3 \
        --lambda-adv 1.0 --lambda-im 1.0 --lambda-l2sp 0.01 \
        --num-workers 4 \
        --save-path checkpoints/dg_cdan_v2_cubdg_${TGT}.pt
done
```

Expected: Art 68.8%, Cartoon 71.8%, Paint 60.5% (avg 67.0%)

### 3. NoPCFG Ablation (no grammar, linear classifier)

```bash
# NoPCFG DG-ERM (same protocol, no grammar)
for TGT in Art Cartoon Paint; do
    python scripts/train_dg_nopcfg.py \
        --dataset cubdg --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --pretrained \
        --n-primitives 8 --randaugment --label-smoothing 0.1 \
        --epochs 50 --batch-size 32 --lr 1e-3 --num-workers 4 \
        --save-path checkpoints/dg_erm_v2_nopcfg_cubdg_${TGT}.pt
done

# NoPCFG + CDAN adaptation
for TGT in Art Cartoon Paint; do
    python scripts/adapt_cdan_nopcfg.py \
        --checkpoint checkpoints/dg_erm_v2_nopcfg_cubdg_${TGT}.pt \
        --dataset cubdg --source Photo --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --n-primitives 8 \
        --align-level backbone \
        --epochs 20 --batch-size 32 --lr 1e-4 --lr-disc 1e-3 \
        --lambda-adv 1.0 --lambda-im 1.0 --lambda-l2sp 0.01 \
        --num-workers 4 \
        --save-path checkpoints/dg_cdan_nopcfg_cubdg_${TGT}.pt
done
```

Expected: Art 52.4%, Cartoon 56.2%, Paint 45.7% (avg 51.4%)

### 4. DG-Adversarial Ablation (3-way DANN alignment)

```bash
for TGT in Art Cartoon Paint; do
    python scripts/train_dg.py \
        --dataset cubdg --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --pretrained \
        --n-primitives 8 --max-depth 1 --use-sparsemax --grammar-l1 0.01 \
        --randaugment --label-smoothing 0.1 \
        --adversarial --lambda-adv 0.1 --lr-disc 1e-3 --align-level backbone \
        --epochs 50 --batch-size 32 --lr 1e-3 --num-workers 4 \
        --save-path checkpoints/dg_adv_v2_pcfg_cubdg_${TGT}.pt
done
```

Expected: Art 57.4%, Cartoon 61.1%, Paint 36.0% (avg 51.5%, -5.5pp vs ERM)

### 5. Deeper Grammar Ablation (max_depth=2)

```bash
for TGT in Art Cartoon Paint; do
    python scripts/train_dg.py \
        --dataset cubdg --target $TGT \
        --data-root ./data/cub/CUB-DG --backbone resnet50 --pretrained \
        --n-primitives 8 --max-depth 2 --use-sparsemax --grammar-l1 0.01 \
        --randaugment --label-smoothing 0.1 \
        --epochs 50 --batch-size 32 --lr 1e-3 --num-workers 4 \
        --save-path checkpoints/dg_erm_v2_depth2_pcfg_cubdg_${TGT}.pt
done
```

Expected: Art 56.1%, Cartoon 62.1%, Paint 40.0% (avg 52.7%, -4.2pp vs depth-1)

### Pretrained Checkpoints (HuggingFace)

All checkpoints are hosted at [`datvo06/neurosymbolic-da-results`](https://huggingface.co/datvo06/neurosymbolic-da-results).

| Checkpoint | Target | Acc | Description |
|------------|--------|-----|-------------|
| `dg_cdan_v2_cubdg_Art.pt` | Art | 68.8% | Best: PCFG + CDAN |
| `dg_cdan_v2_cubdg_Cartoon.pt` | Cartoon | 71.8% | Best: PCFG + CDAN |
| `dg_cdan_v2_cubdg_Paint.pt` | Paint | 60.5% | Best: PCFG + CDAN |
| `dg_cdan_v2_cubdg_Photo.pt` | Photo | 74.3% | Best: PCFG + CDAN |
| `dg_erm_v2_pcfg_cubdg_Art.pt` | Art | 59.9% | DG-ERM (pre-CDAN) |
| `dg_erm_v2_pcfg_cubdg_Cartoon.pt` | Cartoon | 66.5% | DG-ERM (pre-CDAN) |
| `dg_erm_v2_pcfg_cubdg_Paint.pt` | Paint | 44.3% | DG-ERM (pre-CDAN) |
| `dg_erm_v2_pcfg_cubdg_Photo.pt` | Photo | 73.6% | DG-ERM (pre-CDAN) |
| `dg_cdan_nopcfg_cubdg_*.pt` | All 4 | 52.9% avg | NoPCFG ablation |
| `dg_adv_v2_pcfg_cubdg_*.pt` | 3 tgt | 51.5% avg | Adversarial ablation |
| `dg_erm_v2_depth2_pcfg_cubdg_*.pt` | 3 tgt | 52.7% avg | Depth-2 ablation |

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
repo = 'datvo06/neurosymbolic-da-results'

# Best models (PCFG + CDAN, all 4 targets)
for target in ['Art', 'Cartoon', 'Paint', 'Photo']:
    hf_hub_download(repo, f'checkpoints/dg_cdan_v2_cubdg_{target}.pt', local_dir='.')

# DG-ERM source models (pre-adaptation)
for target in ['Art', 'Cartoon', 'Paint', 'Photo']:
    hf_hub_download(repo, f'checkpoints/dg_erm_v2_pcfg_cubdg_{target}.pt', local_dir='.')

# NoPCFG ablation (no grammar)
for target in ['Art', 'Cartoon', 'Paint', 'Photo']:
    hf_hub_download(repo, f'checkpoints/dg_cdan_nopcfg_cubdg_{target}.pt', local_dir='.')

# Adversarial ablation
for target in ['Art', 'Cartoon', 'Paint']:
    hf_hub_download(repo, f'checkpoints/dg_adv_v2_pcfg_cubdg_{target}.pt', local_dir='.')

# Depth-2 ablation
for target in ['Art', 'Cartoon', 'Paint']:
    hf_hub_download(repo, f'checkpoints/dg_erm_v2_depth2_pcfg_cubdg_{target}.pt', local_dir='.')
"
```

### Evaluate a Checkpoint

```bash
python -c "
import torch
from neurosymbolic_da.data.cubdg import get_cubdg
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline
from neurosymbolic_da.training.trainer import evaluate
from torch.utils.data import DataLoader

device = torch.device('cuda')
target = 'Art'  # or 'Cartoon', 'Paint'

model = NeuroSymbolicPipeline(
    n_primitives=8, n_classes=200, backbone_variant='resnet50',
    pretrained_backbone=False, use_sparsemax=True,
)
ckpt = torch.load(f'checkpoints/dg_cdan_v2_cubdg_{target}.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)

tgt_test = get_cubdg('./data/cub/CUB-DG', target, train=False)
loader = DataLoader(tgt_test, batch_size=32, num_workers=4)
loss, acc = evaluate(model, loader, device)
print(f'Target {target}: {acc:.1%}')
"
```

## Project Structure

```
data/
  cub/
  officehome/
  pacs/
  vlcs/
neurosymbolic_da/
  dsl/                    # Layout DSL (effectful algebraic effects)
    ops.py                # 5 DSL operations: has, rel, conj, choice, score
    primitives.py         # Primitive dataclass and Env type
    relations.py          # 6 spatial relations + learnable RelationParams
    grammar.py            # LayoutGrammar (universal PCFG, vectorized eval)
    handlers/             # Handler-based polymorphism
      eval.py             # Direct evaluation -> scalar Tensor
      inside.py           # Inside algorithm -> dict[frozenset, Tensor]
      symbolic.py         # Tree builder -> DerivNode
  nn/                     # Neural network components
    backbone.py           # ResNet feature extractor
    bottleneck.py         # Concept bottleneck (kornia soft-argmax)
    pipeline.py           # Full end-to-end pipeline
    pipeline_nopcfg.py    # Ablation: linear classifier (no grammar)
    pipeline_nobottleneck.py  # Ablation: no bottleneck
  data/                   # Dataset loading
    cubdg.py              # CUB-DG (4 domains, 200 species)
    digits.py             # MNIST, USPS, SVHN
    office.py             # Office-31, Office-Home
    scb.py                # Synthetic Compositional Benchmark
  training/               # Training infrastructure
    trainer.py            # Training loop
    adapt.py              # Adaptation loop
    adversarial.py        # CDAN / DANN adversarial alignment
    losses.py             # MMD, entropy, L2-SP losses
    pmcmc.py              # Particle MCMC for grammar structure search

scripts/
  train_dg.py             # DG training (ERM / adversarial / domain-conditional)
  train_source.py         # Single-source training
  adapt_target.py         # Unsupervised adaptation (Phase 2)
  train_nopcfg.py         # NoPCFG ablation
  extract_derivations.py  # Extract interpretable grammar trees

tests/                    # 170+ unit tests
```

## Tests

```bash
uv run pytest -v
```

## Hardware

All experiments run on a single NVIDIA A40 GPU (46GB). DG-ERM training: ~4 hours (50 epochs). CDAN adaptation: ~90 min (20 epochs).

## Citation

If you use this code, please cite:

```bibtex
@article{nguyen2026neurosymbolic,
  title={Neuro-Symbolic Domain Generalization via Compositional Layout Grammars},
  author={Nguyen, Dat and Nguyen, Duy},
  year={2026}
}
```
