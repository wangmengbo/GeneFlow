#!/bin/bash

# GeneFlow Evaluation Script
# Evaluate trained model on test data

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
BATCH_SIZE=8
GEN_STEPS=50

# Biological evaluation models (optional)
UNI2H_MODEL_PATH=""  # Path to UNI2-h model if available
HE2RNA_WEIGHTS=""    # Path to HE2RNA weights if available

# Advanced options
USE_AMP=""           # Add --use_amp to enable automatic mixed precision
USE_DDP=""           # Add --use_ddp to enable distributed evaluation
SAVE_EMBEDDINGS=""   # Add --save_embeddings to save UNI2-h embeddings for UMAP
EMBEDDINGS_DIR=""    # Add --embeddings_output_path /path/to/embeddings to specify custom output path

# Run evaluation
python rectified/rectified_evaluate.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gen_steps ${GEN_STEPS} \
    ${USE_AMP} \
    ${USE_DDP} \
    ${SAVE_EMBEDDINGS} \
    ${EMBEDDINGS_DIR}