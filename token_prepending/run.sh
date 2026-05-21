#!/bin/bash
MODEL_CONFIG=${1:-"mac-qwen2.5-7b-vi-tp"}
CONFIG_FILE=${2:-"config.yaml"}

echo "using model: $MODEL_CONFIG"
echo "using config file: $CONFIG_FILE"
echo "==============="

python evaluate.py --config "$MODEL_CONFIG" --config_file "$CONFIG_FILE"
