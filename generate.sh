#!/bin/bash

# GeneFlow Image Generation Script
# Generate synthetic histopathological images from gene expression

# Model path
MODEL_PATH="/GeneFlow/results/checkpoints/best_model.pt"

# Data paths
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/generated_results"

# Model configuration
MODEL_TYPE="single"
IMG_SIZE=256
IMG_CHANNELS=4

# Generation parameters
BATCH_SIZE=8
NUM_SAMPLES=100
GEN_STEPS=50

# Stain normalization (optional)
ENABLE_STAIN_NORM=""          # Add --enable_stain_normalization to enable
STAIN_NORM_METHOD="skimage_hist_match"  # Method for stain normalization

# Run generation
python rectified/rectified_generate.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --gen_steps ${GEN_STEPS} \
    ${ENABLE_STAIN_NORM}