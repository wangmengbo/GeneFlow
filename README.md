# GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow
![cover](misc/genecover.png "Cover")

## Overview
![overview](misc/geneflow-cr.png "GeneFlow Overview")

## Requirement
Please refer to the `requirements.txt` file for the list of required packages and their versions. Dependices can be installed using pip:
```bash
pip install -r requirements.txt
```

For evaluation with biological features, please install the additional dependencies, UNI2 and HE2RNA.

## Usage
Preprocessed data of three demo 10x Xenium samples can be found 
To run the GeneFlow model, use the following command:
<!-- ```bash
tar -czf processed_data.tar.gz \
  --transform='s,^depot/natallah/data/Mengbo/HnE_RNA/GeneFlow/,,' \
  -C /depot/natallah/data/Mengbo/HnE_RNA/GeneFlow \
  processed_data/Xenium_Prime_Human_Skin_FFPE/cell_patch_256_aux/input \
  processed_data/Xenium_Prime_Human_Skin_FFPE/adata.h5ad \
  processed_data/Xenium_Prime_Human_Skin_FFPE/adata_unfiltered.h5ad \
  processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input \
  processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad \
  processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata_unfiltered.h5ad \
  processed_data/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/cell_patch_256_aux/input \
  processed_data/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/adata.h5ad \
  processed_data/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/adata_unfiltered.h5ad
``` -->

```bash
python rectified/rectified_main.py \
	--model_type single \
	--adata <path to adata.h5ad> \
	--image_paths <path to cell_image_paths.json> \
	--img_size 256 \
	--img_channels 4 \
	--output_dir $output_dir \
	--batch_size 32 \
	--epochs 50 \
	--patience 5

python rectified/rectified_main.py \
	--model_type multi \
	--adata <path to adata_unfiltered.h5ad> \
	--patch_image_paths <path to patch_image_paths.json> \
	--patch_cell_mapping <path to patch_cell_mapping.json> \
	--img_size 256 \
	--img_channels 4 \
	--output_dir <output_dir> \
	--batch_size 32 \
	--epochs 50 \
	--patience 5
```

Proprocessing code for Xenium data in HEST-1k can be found in `utils/prepare_hestxenium_data.py`.

## Citation
If you find this code useful for your research, please cite the following paper:

