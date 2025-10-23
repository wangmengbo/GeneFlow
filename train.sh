python rectified/rectified_main.py \
	--model_type single \
	--adata '/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad' \
	--image_paths '/GeneFlow/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json' \
	--img_size 256 \
	--img_channels 4 \
	--output_dir '/GeneFlow/results' \
	--batch_size 16 \
	--epochs 50 \
	--patience 5