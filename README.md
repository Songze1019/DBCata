# DBCata

[](https://www.python.org/downloads/release/python-3120/)
[](https://pytorch.org/)

Official implementation of the paper: **"Accelerating High-Throughput Catalyst Screening by Direct Generation of Equilibrium Adsorption Structures"**.

![DBCata](assets/TOC_.drawio.png)

## 🛠️ Installation

This project requires **Python 3.12** and **CUDA 12.4**. The core dependencies include:

  - [PyTorch](https://pytorch.org/)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
  - [PyTorch Lightning](https://lightning.ai/)

We strongly recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### Option 1: Step-by-Step Installation (Recommended)

```bash
# 1. Create and activate a virtual environment
uv venv ~/.dbcata
uv activate ~/.dbcata

# 2. Install basic dependencies
uv pip install lightning==2.2.5 fairchem-core pymatgen jupyter scikit-learn py3Dmol torch_geometric

# 3. Install PyTorch (CUDA 12.4 support)
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 

# 4. Install PyTorch Geometric dependencies
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### Option 2: Install via `requirements.txt`

```bash
uv pip install -r requirements.txt
```

## 📂 Data & Checkpoints

Before training or inference, please download the necessary datasets and pre-trained models (if needed):

| Resource | Description | Link | Path |
| :--- | :--- | :--- | :--- |
| **Dataset** | Pickled lists of `torch_geometric.data.Data` objects (e.g., `train.pkl`, `val.pkl`). | [Download](https://doi.org/10.6084/m9.figshare.30882545) | `data/` (e.g., `data/cathub/`) |
| **Checkpoints** | Pre-trained model weights. | [Download](https://doi.org/10.6084/m9.figshare.30882545) | `checkpoints/` |

## 🚀 Quick Start

We provide a Jupyter Notebook to demonstrate the capabilities of DBCata:

  - **`notebook/MLIP.ipynb`**: A quick demo comparing DBCata with MLIP for adsorption structure prediction.

The UMA pre-trained model checkpoint is required for running the notebook.

[UMA-s(m)-1.1 Permission Application](https://huggingface.co/facebook/UMA)

## 🏋️ Training

### 1\. Generative Model (Main)

To train the structure generation model:

```bash
# Option A: Using SLURM
sbatch train.sh

# Option B: Direct Python execution
python -m scripts.train
```

### 2\. Detection Model (Classifier)

To train the auxiliary classification model:

```bash
# Option A: Using SLURM
sbatch train_bc.sh

# Option B: Direct Python execution
python -m scripts.train_bcmodel
```

-----
