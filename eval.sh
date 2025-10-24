#!/bin/bash

# GeneFlow Evaluation Script
# Modify paths and parameters according to your trained model and data

# Model and data paths
MODEL_DIR="/GeneFlow/results"  # Directory containing trained model checkpoints
ADATA="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad"
IMAGE_PATHS="/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json"
OUTPUT_DIR="/GeneFlow/evaluation_results"

# Model configuration (must match training)
MODEL_TYPE="single"
IMG_SIZE=256
IMG_CHANNELS=4

# Evaluation parameters
BATCH_SIZE=16
GEN_STEPS=50  # Number of ODE solver steps for generation

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run evaluation (generation + gene importance analysis)
python rectified/rectified_main.py \
    --model_type ${MODEL_TYPE} \
    --adata ${ADATA} \
    --image_paths ${IMAGE_PATHS} \
    --img_size ${IMG_SIZE} \
    --img_channels ${IMG_CHANNELS} \
    --output_dir ${MODEL_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs 0 \
    --resume_from ${MODEL_DIR}/checkpoints/best_checkpoint.pt \
    --gen_steps ${GEN_STEPS}

# Copy evaluation results to dedicated directory
echo "Copying evaluation results..."
cp -r ${MODEL_DIR}/generated_images ${OUTPUT_DIR}/
cp ${MODEL_DIR}/generation_results.png ${OUTPUT_DIR}/
cp ${MODEL_DIR}/gene_importance_scores.csv ${OUTPUT_DIR}/ 2>/dev/null || echo "Gene importance analysis not available for multi-cell models"

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"

# Optional: Run biological feature evaluation if tools are installed
if command -v python &> /dev/null; then
    if python -c "import sequoia" 2>/dev/null; then
        echo "Running biological feature evaluation with Sequoia..."
        python eval/evaluate_biological_features.py \
            --generated_images ${OUTPUT_DIR}/generated_images \
            --real_images ${IMAGE_PATHS} \
            --output_dir ${OUTPUT_DIR}/biological_features
    else
        echo "Sequoia not installed. Skipping biological feature evaluation."
        echo "Install via: pip install sequoia-pub"
    fi
fi