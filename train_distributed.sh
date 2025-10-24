#!/bin/bash

# GeneFlow Distributed Training Script (Multi-GPU)
# Requires multiple GPUs and torchrun

# Data paths
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/results"

# Model configuration
MODEL_TYPE="single"
IMG_SIZE=256
IMG_CHANNELS=4

# Training parameters
BATCH_SIZE=16  # Per-GPU batch size
EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
PATIENCE=5

# Multi-GPU configuration
NUM_GPUS=8  # Number of GPUs to use

# Run distributed training
torchrun --nproc_per_node=${NUM_GPUS} rectified/rectified_main.py \
    --use_ddp \
    --use_amp \
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
    --patience ${PATIENCE}