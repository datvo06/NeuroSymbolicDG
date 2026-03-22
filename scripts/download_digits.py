#!/usr/bin/env python3
"""Download digit datasets (MNIST, USPS, SVHN) via torchvision."""

import argparse

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="Download digit datasets")
    parser.add_argument("--root", default="./data", help="Download directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "usps", "svhn"],
        choices=["mnist", "usps", "svhn"],
        help="Which datasets to download",
    )
    args = parser.parse_args()

    for name in args.datasets:
        print(f"Downloading {name}...")
        if name == "mnist":
            datasets.MNIST(args.root, train=True, download=True)
            datasets.MNIST(args.root, train=False, download=True)
        elif name == "usps":
            datasets.USPS(args.root, train=True, download=True)
            datasets.USPS(args.root, train=False, download=True)
        elif name == "svhn":
            datasets.SVHN(args.root, split="train", download=True)
            datasets.SVHN(args.root, split="test", download=True)
        print(f"  {name} done.")

    print(f"\nAll datasets saved to {args.root}/")


if __name__ == "__main__":
    main()
