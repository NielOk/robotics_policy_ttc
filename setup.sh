#!/bin/bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download dataset
# Exit immediately on error
set -e

# Define the tag and repo
REPO="NielOk/robotics_policy_ttc"
TAG="dataset_v1"
ASSET_NAME="pusht_cchi_v7_replay.zip"

# Create target directory
TARGET_DIR="./"
mkdir -p "$TARGET_DIR"

# Download the dataset
echo "Downloading dataset from GitHub release..."
curl -L -o "$TARGET_DIR/$ASSET_NAME" "https://github.com/$REPO/releases/download/$TAG/$ASSET_NAME"

# Unzip the dataset
echo "Extracting dataset..."
unzip -q "$TARGET_DIR/$ASSET_NAME" -d "$TARGET_DIR"
rm "$TARGET_DIR/$ASSET_NAME"

# Done
echo "Dataset installed at: $TARGET_DIR"