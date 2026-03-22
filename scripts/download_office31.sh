#!/usr/bin/env bash
# Download Office-31 dataset.
#
# Usage: bash scripts/download_office31.sh [TARGET_DIR]
#
# This downloads from a common mirror. If it fails, manually download from:
#   https://www.kaggle.com/datasets/xixuhu/office31
# and extract to TARGET_DIR/

set -euo pipefail

TARGET_DIR="${1:-./data/office31}"
mkdir -p "$TARGET_DIR"

echo "=== Office-31 Dataset ==="
echo "Target directory: $TARGET_DIR"

# Check if already downloaded
if [ -d "$TARGET_DIR/amazon" ] && [ -d "$TARGET_DIR/dslr" ] && [ -d "$TARGET_DIR/webcam" ]; then
    echo "Office-31 already exists at $TARGET_DIR. Skipping download."
    exit 0
fi

echo ""
echo "Office-31 requires manual download due to license restrictions."
echo ""
echo "Option 1: Kaggle"
echo "  1. Visit: https://www.kaggle.com/datasets/xixuhu/office31"
echo "  2. Download and extract to: $TARGET_DIR/"
echo ""
echo "Option 2: Kaggle CLI"
echo "  pip install kaggle"
echo "  kaggle datasets download -d xixuhu/office31 -p $TARGET_DIR --unzip"
echo ""
echo "Expected layout after extraction:"
echo "  $TARGET_DIR/amazon/images/<class_name>/*.jpg"
echo "  $TARGET_DIR/dslr/images/<class_name>/*.jpg"
echo "  $TARGET_DIR/webcam/images/<class_name>/*.jpg"
echo ""

# Try kaggle CLI if available
if command -v kaggle &> /dev/null; then
    echo "Kaggle CLI found. Attempting download..."
    kaggle datasets download -d xixuhu/office31 -p "$TARGET_DIR" --unzip
    echo "Office-31 downloaded successfully to $TARGET_DIR/"
else
    echo "Kaggle CLI not found. Please install with: pip install kaggle"
    echo "Then re-run this script, or download manually."
    exit 1
fi
