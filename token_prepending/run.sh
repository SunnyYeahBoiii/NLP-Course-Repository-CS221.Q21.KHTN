#!/bin/bash
MODEL_CONFIG=${1:-"llama-2-7b"}
CONFIG_FILE=${2:-"config.yaml"}
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    shift
fi

echo "using model: $MODEL_CONFIG"
echo "using config file: $CONFIG_FILE"
echo "==============="

python evaluate.py --config "$MODEL_CONFIG" --config_file "$CONFIG_FILE" "$@"
