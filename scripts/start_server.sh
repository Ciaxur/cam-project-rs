#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname $0)"
PROJ_DIR="$SCRIPT_DIR/.."

# Check if virtual env already exist.
[ -d "opencv-venv" ] || { echo "Virtual Environment does not exist, creating one"; python3 -m venv opencv-venv; }

# Install python packages.
. opencv-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Start server.
python3 src/server.py \
  --threshold=0.7

