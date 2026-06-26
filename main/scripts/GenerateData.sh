#!/bin/bash

DATASET_NAME='WILL'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../..")"/

# Training Dataset
python data_generator/GeneratorPatchData.py \
    --N 4 \
    --sampler-dir data_generator/sampled_data/data_WILL/ \
    --output-dir train_data/ \
    --logger log_train_data.txt \