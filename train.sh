#!/bin/bash

# GeneFlow Training Script
# Modify paths and parameters according to your data and requirements

# Data paths
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/results"

# Model configuration
MODEL_TYPE="single"  # Options: single, multi
IMG_SIZE=256
IMG_CHANNELS=4

# Training parameters
BATCH_SIZE=16
EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
PATIENCE=5

# Advanced options
USE_AMP=""  # Add --use_amp to enable automatic mixed precision
USE_DDP=""  # Add --use_ddp to enable distributed data parallel training

# Run training
python rectified/rectified_main.py \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    ${USE_AMP} \
    ${USE_DDP}