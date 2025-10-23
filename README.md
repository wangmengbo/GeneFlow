# GeneFlow

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/114997)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org/)

Official implementation of **GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow** (NeurIPS 2025).

[Paper](https://neurips.cc/virtual/2025/poster/114997) | [Citation](#citation)

![cover](misc/genecover.png "Cover")

## Architecture

![overview](misc/geneflow-cr.png "GeneFlow Overview")

---

## Installation

- CUDA 12.1 compatible GPU
- Conda or Miniconda

1. **Create conda environment with Python 3.11:**
```bash
conda create -n geneflow python=3.11
conda activate geneflow
```

2. **Install PyTorch with CUDA 12.1:**
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Data Download and Preprocessing

Preprocessing code for Xenium data in HEST-1k is provided in `utils/prepare_hestxenium_data.py`.

---

## Training

```bash
bash train.sh
```
---

## Evaluation

For evaluation using biological features, install additional dependencies:

### Required Tools
- [Sequoia](https://github.com/gevaertlab/sequoia-pub)
- [UNI2](https://huggingface.co/MahmoodLab/UNI2-h)
- [Pretrained HE2RNA weights](https://huggingface.co/gevaertlab)

### Setup Instructions

1. **Download pretrained HE2RNA weights** and place them in the `sequoia/models/` directory.

2. **Modify `sequoia/src/he2rna.py`** by replacing the import statements:

```python
# Replace these lines:
# from src.read_data import SuperTileRNADataset
# from src.utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features

# With:
from .read_data import SuperTileRNADataset
from .utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features
```

---

## Citation

If you find this work useful for your research, please cite:

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

---

## Contact

For questions or issues, please open an issue on this repository.