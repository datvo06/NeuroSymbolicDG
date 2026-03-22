#!/usr/bin/env bash
# Download Office-Home dataset.
#
# Usage: bash scripts/download_officehome.sh [TARGET_DIR]
#
# Office-Home is available from the official project page.
# If automatic download fails, manually download from:
#   https://www.hemanthdv.org/OfficeHome-Dataset/

set -euo pipefail

TARGET_DIR="${1:-./data/officehome}"
mkdir -p "$TARGET_DIR"

echo "=== Office-Home Dataset ==="
echo "Target directory: $TARGET_DIR"

# Check if already downloaded
if [ -d "$TARGET_DIR/Art" ] && [ -d "$TARGET_DIR/Clipart" ] && \
   [ -d "$TARGET_DIR/Product" ] && [ -d "$TARGET_DIR/Real_World" ]; then
    echo "Office-Home already exists at $TARGET_DIR. Skipping download."
    exit 0
fi

echo ""
echo "Office-Home requires manual download."
echo ""
echo "Option 1: Official website"
echo "  1. Visit: https://www.hemanthdv.org/OfficeHome-Dataset/"
echo "  2. Download the dataset"
echo "  3. Extract to: $TARGET_DIR/"
echo ""
echo "Option 2: Hugging Face"
echo "  pip install huggingface_hub"
echo "  python -c \""
echo "    from huggingface_hub import snapshot_download"
echo "    snapshot_download('flwrlabs/office-home', local_dir='$TARGET_DIR', repo_type='dataset')"
echo "  \""
echo ""
echo "Expected layout after extraction:"
echo "  $TARGET_DIR/Art/<class_name>/*.jpg"
echo "  $TARGET_DIR/Clipart/<class_name>/*.jpg"
echo "  $TARGET_DIR/Product/<class_name>/*.jpg"
echo "  $TARGET_DIR/Real_World/<class_name>/*.jpg"
echo ""

# Try huggingface_hub if available
if python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "huggingface_hub found. Attempting download..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('flwrlabs/office-home', local_dir='$TARGET_DIR', repo_type='dataset')
"
    echo "Office-Home downloaded successfully to $TARGET_DIR/"
else
    echo "Please download manually using one of the options above."
    exit 1
fi
