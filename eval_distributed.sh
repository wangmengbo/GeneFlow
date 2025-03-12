#!/bin/bash

# GeneFlow Distributed Evaluation Script (Multi-GPU)
# Requires multiple GPUs and torchrun

# Model path
MODEL_PATH="/GeneFlow/results/checkpoints/best_model.pt"

# Data paths
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/evaluation_results"

# Model configuration
MODEL_TYPE="single"
IMG_SIZE=256
IMG_CHANNELS=4

# Evaluation parameters
BATCH_SIZE=8  # Per-GPU batch size
GEN_STEPS=50

# Multi-GPU configuration
NUM_GPUS=8  # Number of GPUs to use

# Run distributed evaluation
torchrun --nproc_per_node=${NUM_GPUS} rectified/rectified_evaluate.py \
    --use_ddp \
    --use_amp \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gen_steps ${GEN_STEPS}