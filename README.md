# GeneFlow

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/114997)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org/)

Official implementation of **GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow** (NeurIPS 2025).

[Paper](https://neurips.cc/virtual/2025/poster/114997) | [Citation](#citation)

![cover](misc/genecover.png "Cover")

---

## News

- **October 2025**: Initial release of GeneFlow codebase
- **October 2025**: Preprocessed demo Xenium samples released on Zenodo
- **September 2025**: Paper accepted at NeurIPS 2025

---

## Overview

![overview](misc/geneflow-cr.png "GeneFlow Overview")

GeneFlow translates single-cell gene expression profiles to histopathological images using rectified flow models. This approach enables the generation of synthetic tissue images from spatial transcriptomics data, facilitating downstream analyses in computational pathology and spatial biology.

---

## Installation

### Requirements

- CUDA 12.1 compatible GPU (tested on 8×H100 with DDP and 1×H100)
- Conda or Miniconda
- Python 3.11

### Setup

1. **Create conda environment:**
```bash
conda create -n geneflow python=3.11
conda activate geneflow
```

2. **Install PyTorch with CUDA 12.1:**
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Data

### Download Preprocessed Demo Data

We provide two preprocessed Xenium samples for demonstration:

**Sample C1:**
```bash
wget -O xenium_c1.tar.gz "https://zenodo.org/records/17429425/files/xenium_c1.tar.gz?download=1"
tar -xzf xenium_c1.tar.gz -C /path/to/GeneFlow/data/
```

**Sample C2:**
```bash
wget -O xenium_c2.tar.gz "https://zenodo.org/records/17429434/files/xenium_c2.tar.gz?download=1"
tar -xzf xenium_c2.tar.gz -C /path/to/GeneFlow/data/
```

### Data Preprocessing

For preprocessing custom Xenium data from HEST-1k:

```bash
python utils/prepare_hestxenium_data.py \
    --input_dir /path/to/raw/data \
    --output_dir /path/to/processed/data \
    --img_size 256 \
    --img_channels 4
```

See `utils/README.md` for detailed preprocessing instructions.

---

## Training

### Quick Start

Run training with default parameters:

```bash
bash train.sh
```

### Custom Training

Modify `train.sh` or run directly with custom arguments:

```bash
python rectified/rectified_main.py \
    --model_type single \
    --adata /path/to/adata.h5ad \
    --image_paths /path/to/cell_image_paths.json \
    --img_size 256 \
    --img_channels 4 \
    --output_dir /path/to/results \
    --batch_size 16 \
    --epochs 50 \
    --patience 5 \
    --lr 1e-4 \
    --use_amp
```

### Distributed Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=8 rectified/rectified_main.py \
    --use_ddp \
    --model_type single \
    --adata /path/to/adata.h5ad \
    --image_paths /path/to/cell_image_paths.json \
    --img_size 256 \
    --img_channels 4 \
    --output_dir /path/to/results \
    --batch_size 16 \
    --epochs 50
```

---

## Evaluation

### Quick Start

Run evaluation on trained model:

```bash
bash eval.sh
```

### Biological Feature Evaluation

For comprehensive evaluation using biological features:

#### Required Tools

1. **Sequoia**: Cell segmentation and feature extraction
   - Repository: [https://github.com/gevaertlab/sequoia-pub](https://github.com/gevaertlab/sequoia-pub)
   - Installation: `pip install sequoia-pub`

2. **UNI2**: Universal histopathology foundation model
   - Model: [https://huggingface.co/MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)

3. **HE2RNA Weights**: Pretrained histology-to-transcriptomics model
   - Weights: [https://huggingface.co/gevaertlab](https://huggingface.co/gevaertlab)

#### Setup Instructions

1. **Download HE2RNA weights:**
```bash
wget -O sequoia/models/he2rna_weights.pt "https://huggingface.co/gevaertlab/he2rna/resolve/main/weights.pt"
```

2. **Modify Sequoia imports** in `sequoia/src/he2rna.py`:
```python
# Replace:
# from src.read_data import SuperTileRNADataset
# from src.utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features

# With:
from .read_data import SuperTileRNADataset
from .utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features
```

3. **Run evaluation:**
```bash
python eval/evaluate_biological_features.py \
    --generated_images /path/to/generated_images \
    --real_images /path/to/real_images \
    --output_dir /path/to/eval_results
```

---

## Model Architecture

GeneFlow employs a rectified flow framework to map gene expression profiles to histopathological images:

- **Input**: Single-cell or multi-cell gene expression vectors
- **Encoder**: Transformer-based architecture for processing gene expression
- **Flow Model**: Rectified flow for continuous mapping between distributions
- **Output**: Synthetic histopathological images (RGB + auxiliary channels)

Key features:
- Supports single-cell and multi-cell (patch-level) generation
- Configurable number of image channels (RGB + auxiliary)
- Optional spatial graph loss for multi-cell coherence
- Mixed precision training with automatic mixed precision (AMP)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{geneflow2025,
  title={GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow},
  author={},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).

[![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-nd/4.0/)

**Permitted:**
- Use for academic and research purposes
- Citation in academic publications

**Prohibited:**
- Commercial use
- Distribution of modified versions
- Use in production systems without explicit permission

---

## Contact

For questions, issues, or collaboration inquiries:

- Open an issue on this repository
- Check existing issues before creating new ones
- Provide detailed information for bug reports (OS, CUDA version, error messages)

---

## Acknowledgments

We thank the developers of HEST-1k, Xenium platform, and the computational pathology community for their foundational contributions to spatial biology.