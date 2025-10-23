# GeneFlow
![cover](misc/genecover.png "Cover")

## Overview
![overview](misc/geneflow-cr.png "GeneFlow Overview")
This is the repo for the NeurIPS 2025 paper, [GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow](https://neurips.cc/virtual/2025/poster/114997).

## Requirement
Please refer to the `requirements.txt` file for the list of required packages and their versions. Dependices can be installed using pip:
```bash
pip install -r requirements.txt
```

## Usage
Preprocessed data of three demo 10x Xenium samples can be found 
To run the GeneFlow model, use the following commands:
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

For evaluation with biological features, please install the additional dependencies, including [Sequoia](https://github.com/gevaertlab/sequoia-pub), [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) and [pretrained HE2RNA weights](https://huggingface.co/gevaertlab). Please download the pretrained HE2RNA weights and place them in the `sequoia/models/` directory. Replce these lines in `sequoia/src/he2rna.py` with provided code:
```python
# from src.read_data import SuperTileRNADataset
# from src.utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features
from .read_data import SuperTileRNADataset
from .utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features
```


<!-- ## Citation
If you find this code useful for your research, please cite the following paper: -->

## License
Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg