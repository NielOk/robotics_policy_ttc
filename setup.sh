#!/bin/bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download dataset
REPO="NielOk/robotics_policy_ttc"
TAG="dataset_v1"
DATA_ASSET_NAME="pusht_cchi_v7_replay.zip"
DATA_TARGET_DIR="./"

echo "Downloading dataset from GitHub release..."
curl -L -o "$DATA_TARGET_DIR/$DATA_ASSET_NAME" "https://github.com/$REPO/releases/download/$TAG/$DATA_ASSET_NAME"

# Unzip the dataset
echo "Extracting dataset..."
unzip -q "$DATA_TARGET_DIR/$DATA_ASSET_NAME" -d "$DATA_TARGET_DIR"
rm "$DATA_TARGET_DIR/$DATA_ASSET_NAME"

# Done
echo "Dataset installed at: $DATA_TARGET_DIR"

# Download pretrained statebased 100 epoch model
STATE_100_EPOCH_MODEL="pusht_state_100ep.ckpt"
STATE_100_EPOCH_MODEL_TARGET_DIR="./push_t_state_policy_experiments/"

echo "Downloading 100 epoch chkpt statebased model from GitHub release..."
curl -L -o "$STATE_100_EPOCH_MODEL_TARGET_DIR/$STATE_100_EPOCH_MODEL" "https://github.com/NielOk/robotics_policy_ttc/releases/download/$TAG/$STATE_100_EPOCH_MODEL"\