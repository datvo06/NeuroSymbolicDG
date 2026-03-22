#!/usr/bin/env python3
"""Extract top derivation trees for interpretability (Experiment 3, Section 6.5).

Loads a trained checkpoint and uses the symbolic handler to extract
the highest-weighted derivation structures per class.

Usage:
    uv run python scripts/extract_derivations.py \
        --checkpoint checkpoint_digits_mnist_usps.pt \
        --n-primitives 8 --n-classes 10 --top-k 3
"""

import argparse

import torch
from effectful.ops.semantics import handler
from torch import Tensor

from neurosymbolic_da.dsl.grammar import LayoutGrammar
from neurosymbolic_da.dsl.handlers.symbolic import DerivNode, make_symbolic_handler
from neurosymbolic_da.dsl.relations import RELATION_NAMES
from neurosymbolic_da.nn.pipeline import NeuroSymbolicPipeline


def get_top_productions(
    grammar: LayoutGrammar, class_idx: int, top_k: int = 3
) -> list[tuple[float, dict]]:
    """Get the top-k highest-weighted productions for a class.

    Returns:
        List of (weight, production_dict) sorted by weight descending.
    """
    weights = torch.softmax(grammar.log_weights[class_idx], dim=0)
    indexed = list(enumerate(weights.tolist()))
    indexed.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, w in indexed[:top_k]:
        prod = grammar.productions[idx]
        results.append((w, prod))
    return results


def format_production(prod: dict) -> str:
    """Format a production dict as a readable string."""
    if prod["type"] == "has":
        return f"has(p{prod['prim']})"
    else:
        return f"{prod['name']}(p{prod['a']}, p{prod['b']})"


def extract_symbolic_tree(grammar: LayoutGrammar, class_idx: int) -> DerivNode:
    """Build the full symbolic derivation tree for a class."""
    with handler(make_symbolic_handler()):
        tree = grammar(class_idx)
    return tree


def print_class_summary(
    grammar: LayoutGrammar, class_idx: int, top_k: int = 3, class_name: str | None = None
) -> None:
    """Print a summary of a class's grammar structure."""
    label = class_name or str(class_idx)
    print(f"\n{'='*60}")
    print(f"Class {label}")
    print(f"{'='*60}")

    # Top-k productions by weight
    top_prods = get_top_productions(grammar, class_idx, top_k)
    print(f"\nTop-{top_k} productions (by weight):")
    for rank, (w, prod) in enumerate(top_prods, 1):
        print(f"  {rank}. {format_production(prod)}  (weight={w:.4f})")

    # Active productions (weight > uniform baseline)
    n_prods = grammar.n_productions
    uniform = 1.0 / n_prods
    weights = torch.softmax(grammar.log_weights[class_idx], dim=0)
    active = [(i, w.item()) for i, w in enumerate(weights) if w.item() > 2 * uniform]
    active.sort(key=lambda x: x[1], reverse=True)

    print(f"\nActive productions (>{2*uniform:.4f} weight): {len(active)}/{n_prods}")
    for idx, w in active[:10]:
        prod = grammar.productions[idx]
        print(f"  {format_production(prod)}  (weight={w:.4f})")
    if len(active) > 10:
        print(f"  ... and {len(active) - 10} more")

    # Relation distribution
    rel_weights: dict[str, float] = {name: 0.0 for name in RELATION_NAMES}
    has_weight = 0.0
    for idx, w in enumerate(weights.tolist()):
        prod = grammar.productions[idx]
        if prod["type"] == "has":
            has_weight += w
        else:
            rel_weights[prod["name"]] += w

    print(f"\nRelation weight distribution:")
    print(f"  has:        {has_weight:.4f}")
    for name in RELATION_NAMES:
        print(f"  {name:12s} {rel_weights[name]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract derivation trees for interpretability"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-primitives", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--class-names", nargs="*", default=None,
        help="Optional class names (e.g., 0 1 2 ... 9 for digits)"
    )
    parser.add_argument("--classes", nargs="*", type=int, default=None,
        help="Specific class indices to show (default: all)")
    parser.add_argument("--show-tree", action="store_true",
        help="Print full symbolic derivation trees")

    args = parser.parse_args()

    # Build model and load checkpoint
    model = NeuroSymbolicPipeline(
        n_primitives=args.n_primitives,
        n_classes=args.n_classes,
        backbone_variant=args.backbone,
        pretrained_backbone=False,
        max_depth=args.max_depth,
        use_inside=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Grammar: {model.grammar.n_productions} productions, "
          f"{args.n_primitives} primitives, {args.n_classes} classes")

    # Print summaries
    class_indices = args.classes or list(range(args.n_classes))
    for c in class_indices:
        name = args.class_names[c] if args.class_names and c < len(args.class_names) else None
        print_class_summary(model.grammar, c, args.top_k, class_name=name)

        if args.show_tree:
            tree = extract_symbolic_tree(model.grammar, c)
            print(f"\nFull derivation tree:")
            print(tree)


if __name__ == "__main__":
    main()
