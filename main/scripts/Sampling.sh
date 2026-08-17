#!/bin/bash

DATASET_NAME='WILL'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../..")"/

# Get Sampler for all labelled data
python data_generator/GeneratorSampler.py \
    --data-root $DATASET_PATH \
    --data-folder real_data/ \
    --dataset-id $DATASET_NAME 
