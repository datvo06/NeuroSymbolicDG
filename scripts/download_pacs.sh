#!/bin/bash
# Download PACS dataset from HuggingFace
# Usage: bash scripts/download_pacs.sh [data_root]
# Default: ./data/pacs

DATA_ROOT="${1:-./data/pacs}"
mkdir -p "$DATA_ROOT"

echo "Downloading PACS dataset to $DATA_ROOT..."
echo "This uses the flwrlabs/pacs dataset from HuggingFace."

# Method 1: Use huggingface_hub to download
python3 -c "
import os, sys
from pathlib import Path

data_root = '$DATA_ROOT'

try:
    from huggingface_hub import snapshot_download
    print('Downloading PACS from HuggingFace...')
    snapshot_download(
        repo_id='flwrlabs/pacs',
        repo_type='dataset',
        local_dir=data_root + '/raw',
    )
    print(f'Downloaded to {data_root}/raw')
    print('Note: HuggingFace version uses Parquet format.')
    print('You may need to extract images to ImageFolder format.')
except ImportError:
    print('huggingface_hub not installed. Trying alternative...')
    print('pip install huggingface_hub')
    sys.exit(1)
"

echo ""
echo "Expected ImageFolder layout:"
echo "  $DATA_ROOT/photo/dog/*.jpg"
echo "  $DATA_ROOT/art_painting/dog/*.jpg"
echo "  $DATA_ROOT/cartoon/dog/*.jpg"
echo "  $DATA_ROOT/sketch/dog/*.jpg"
echo ""
echo "If images are in Parquet format, extract them to ImageFolder layout."
echo "Alternatively, download directly from:"
echo "  https://sketchx.eecs.qmul.ac.uk/downloads/"
echo "  or https://www.kaggle.com/datasets/nickfratto/pacs-dataset"
