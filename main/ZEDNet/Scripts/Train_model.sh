#!/bin/bash

DATASET_PATH="/Users/u5501629/Desktop/Paper/Code_for_paper/sythetic_extrusion_generation/main/train_data/"
EPOCHS=500
DATA_SET_SIZE=100

python ./../Training.py \
	--data-dir "$DATASET_PATH" \
        --epochs "$EPOCHS" \
        --data-set-size "$DATA_SET_SIZE" 
