# SubCellPortable

> **Efficient inference wrapper for the SubCell subcellular protein localization foundation model**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.12.06.627299-blue)](https://www.biorxiv.org/content/10.1101/2024.12.06.627299v2)

SubCellPortable provides a streamlined interface for running the [SubCell model](https://github.com/CellProfiling/subcell-embed) on IF microscopy images. Generate single-cell embeddings that encode cell morphology or protein localization and predict protein subcellular localization from multi-channel fluorescence microscopy images.

**📄 Preprint**: [SubCell: Proteome-aware vision foundation models for microscopy capture single-cell biology](https://www.biorxiv.org/content/10.1101/2024.12.06.627299v2) (Gupta et al., 2024)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/CellProfiling/SubCellPortable.git
cd SubCellPortable

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Image format

SubCell uses single channel IF microscopy images as input. It works at single cell level, so it is recommended that:
- The images show the complete single cell of interest centered
- If the images come from a FOV (in which more cells might be present) and you have the cell segmentation mask, mask the cell with it. This is not mandatory but advised.
- If you are running inference and want to be as close as possible with the best model performance during inference, please use 640x640 images (or more if this size does not cover many of your cells) with a 0.0800885 micron/pixel resolution


### Basic Usage

1. **Prepare your input CSV** (`path_list.csv`):

```csv
r_image,y_image,b_image,g_image,output_prefix
images/cell_1_mt.png,,images/cell_1_nuc.png,images/cell_1_prot.png,cell1_
images/cell_2_mt.png,,images/cell_2_nuc.png,images/cell_2_prot.png,cell2_
```

**Channel mapping:**
- `r` = microtubules (red)
- `y` = ER (yellow)
- `b` = nuclei (blue/DAPI)
- `g` = protein of interest (green)

*Leave channels empty if not available (e.g., use `rbg` for 3-channel images)*

2. **Configure settings** (`config.yaml`):

```yaml
path_list: "path_list.csv"  # The location of the input CSV file
model_channels: "rybg"      # Channel configuration
output_dir: "./results"     # Output directory
batch_size: 128             # Batch size (adjust for GPU memory)
gpu: 0                      # GPU device ID (-1 for CPU)
output_format: "combined"   # "combined" (h5ad) or "individual" (npy)
```

3. **Run inference**:

```bash
python process.py
```

---

## 📖 Usage Guide

### Command-Line Interface

```bash
# Basic run with config file
python process.py

# Specify parameters via CLI
python process.py --output_dir ./results --batch_size 256 --gpu 0

# Custom config and input files
python process.py --config experiment_config.yaml --path_list experiment_data.csv -o ./results

# Embeddings only (faster, no classification)
python process.py -o ./results --embeddings_only

# Get help
python process.py --help
```

### Input CSV Formats

**Recommended Format:**
```csv
r_image,y_image,b_image,g_image,output_prefix
path/to/image1_mt.png,,path/to/image1_nuc.png,path/to/image1_prot.png,sample_1
path/to/image2_mt.png,,path/to/image2_nuc.png,path/to/image2_prot.png,batch_A/sample_2
```

- Skip rows by prefixing with `#`
- Create subfolders in the output folder by them to output_prefix like: /subfolder_1/sample_1

**Legacy Format** (deprecated but supported):
```csv
r_image,y_image,b_image,g_image,output_folder,output_prefix
...
```

---

## ⚙️ Configuration Parameters

### Basic Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--config` | Path to configuration YAML file | `config.yaml` | `experiment.yaml` |
| `--path_list` | Path to input CSV file | `path_list.csv` | `data.csv` |
| `--output_dir` `-o` | Output directory for all results | - | `./results` |
| `--model_channels` `-c` | Channel configuration | `rybg` | `rbg`, `ybg`, `bg` |
| `--model_type` `-m` | Model architecture | `mae_contrast_supcon_model` | `vit_supcon_model` |
| `--output_format` | Output format | `combined` | `individual` |
| `--num_workers` `-w` | Data loading workers | `4` | `8` |
| `--gpu` `-g` | GPU device ID (-1 = CPU) | `-1` | `0` |
| `--batch_size` `-b` | Batch size | `128` | `256` |
| `--embeddings_only` | Skip classification | `False` | - |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--update_model` `-u` | Download/update models | `False` |
| `--prefetch_factor` `-p` | Prefetch batches | `2` |
| `--create_csv` | Generate combined CSV | `False` |
| `--save_attention_maps` | Save attention visualizations | `False` |
| `--async_saving` | Async file saving (individual only) | `False` |
| `--quiet` `-q` | Suppress verbose logging | `False` |

---

## 📦 Output Formats

### Default: Combined H5AD Format

**File**: `embeddings.h5ad` (AnnData-compatible)

```python
import anndata as ad

# Load results
adata = ad.read_h5ad("results/embeddings.h5ad")

# Access data
embeddings = adata.X                    # (n_samples, 1536)
probabilities = adata.obsm['probabilities']  # (n_samples, 31)
sample_ids = adata.obs_names            # Image identifiers
```

**Compatible with**: scanpy and other single-cell tools

### Individual Format

**Files per image**:
- `{output_prefix}_embedding.npy` - 1536D embedding vector
- `{output_prefix}_probabilities.npy` - 31-class probability distribution
- `{output_prefix}_attention_map.png` - Attention visualization (optional)

```python
import numpy as np

embedding = np.load("results/cell1_embedding.npy")      # Shape: (1536,)
probs = np.load("results/cell1_probabilities.npy")      # Shape: (31,)
```

### Optional: Combined CSV

**File**: `result.csv`

| Column | Description |
|--------|-------------|
| `id` | Sample identifier |
| `top_class_name` | Top predicted location |
| `top_class` | Top class index |
| `top_3_classes_names` | Top 3 predictions (comma-separated) |
| `top_3_classes` | Top 3 indices |
| `prob00` - `prob30` | Full probability distribution |
| `feat0000` - `feat1535` | Full embedding vector |

---

## 🎯 Subcellular Location Classes

The model predicts 31 subcellular locations (indexed 0-30, corresponding to `prob00`-`prob30`):

<details>
<summary>View all 31 classes</summary>

00. Actin filaments
01. Aggresome
02. Cell Junctions
03. Centriolar satellite
04. Centrosome
05. Cytokinetic bridge
06. Cytoplasmic bodies
07. Cytosol
08. Endoplasmic reticulum
09. Endosomes
10. Focal adhesion sites
11. Golgi apparatus
12. Intermediate filaments
13. Lipid droplets
14. Lysosomes
15. Microtubules
16. Midbody
17. Mitochondria
18. Mitotic chromosome
19. Mitotic spindle
20. Nuclear bodies
21. Nuclear membrane
22. Nuclear speckles
23. Nucleoli
24. Nucleoli fibrillar center
25. Nucleoli rim
26. Nucleoplasm
27. Peroxisomes
28. Plasma membrane
29. Vesicles
30. Negative

</details>

Class names and visualization colors available in `inference.py` (`CLASS2NAME`, `CLASS2COLOR` dictionaries).

---

## 🔧 Model Setup

### Using Default Models

Models are automatically downloaded on first run with `-u/--update_model`:

```bash
python process.py -u --output_dir ./results
```

### Custom Models

Edit `models_urls.yaml` to specify custom model URLs:

```yaml
rybg:  # 4-channel configuration
  mae_contrast_supcon_model:
    encoder: "s3://bucket/path/to/encoder.pth"
    classifier_s0: "https://url/to/classifier.pth"
```

---

## 🤝 Citation

If you use SubCellPortable in your research, please cite:

```bibtex
@article{gupta2024subcell,
  title={SubCell: Proteome-aware vision foundation models for microscopy capture single-cell biology},
  author={Gupta, Ankit and Wefers, Zoe and Kahnert, Konstantin and Hansen, Jan N. and Leineweber, Will and Cesnik, Anthony and Lu, Dan and Axelsson, Ulrika and Balllosera Navarro, Frederic and Karaletsos, Theofanis and Lundberg, Emma},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.12.06.627299}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/CellProfiling/SubCellPortable/issues)

---

SubCellPortable wrapper maintained with ❤️ by the Lundberg Lab for the computational biology community.
