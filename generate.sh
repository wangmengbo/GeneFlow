#!/bin/bash

# GeneFlow Image Generation Script
# Generate images from gene expression using a pretrained model

# Model and data paths
MODEL_PATH="/GeneFlow/results/checkpoints/best_checkpoint.pt"  # Path to trained model
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/generated_results"

# Model configuration (must match training configuration)
MODEL_TYPE="single"  # Options: single, multi
IMG_SIZE=256
IMG_CHANNELS=4

# Generation parameters
BATCH_SIZE=8
NUM_SAMPLES=100  # Number of samples to generate
GEN_STEPS=50  # Number of ODE solver steps (higher = better quality)

# Stain normalization (optional)
ENABLE_STAIN_NORM=""  # Add --enable_stain_normalization to enable
STAIN_METHOD="skimage_hist_match"  # Options: skimage_hist_match

# Create output directory
mkdir -p ${OUTPUT_DIR}

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
    ${ENABLE_STAIN_NORM} \
    --stain_normalization_method ${STAIN_METHOD}

echo "Generation complete. Results saved to ${OUTPUT_DIR}"
echo "- PDF with all samples: ${OUTPUT_DIR}/generation_results.pdf"
echo "- PNG preview: ${OUTPUT_DIR}/generation_results.png"
echo "- Individual images: ${OUTPUT_DIR}/generated_images/"