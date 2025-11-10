# morpho_integration

This repository contains a PyTorch implementation for **cross-modal integration** of
- a **morphology modality** (e.g. cell morphology features or distance matrices), and  
- a **gene expression modality** (e.g. exon-level expression of selected genes).

The model uses **variational autoencoders (VAEs)** for each modality, an **adversarial discriminator in the shared latent space**, optional **Gromov–Wasserstein (GW) structure alignment**, and an optional **prior loss** based on a precomputed morphology–transcriptomics correlation matrix. fileciteturn1file10turn1file1

---

## 1. Environment & Dependencies

The code is designed to run in a **Linux, CPU-only** environment using **Python 3.8** with a Conda environment named, for example, `cross_modal_cpu_env`:

```bash
conda create -n cross_modal_cpu_env python=3.8
conda activate cross_modal_cpu_env
```

The environment used for the experiments includes the following key packages (from `conda list`):

### Core language and tooling

- `python` 3.8.20  
- `pip` 24.2  
- `setuptools` 75.1.0  
- `wheel` 0.44.0  

### Deep learning and GPU-related (works in CPU-only environment)

- `torch` 2.4.1  
- `torchvision` 0.19.1  
- `torchaudio` 2.4.1  
- `triton` 3.0.0  
- `nvidia-cublas-cu12`, `nvidia-cuda-*`, `nvidia-cudnn-cu12`, etc.  
  > These packages are present in the original environment but the training code also works in a CPU-only setup (CUDA is optional). fileciteturn0file0

### Scientific Python stack

- `numpy` 1.24.4  
- `scipy` 1.10.1  
- `pandas` 2.0.3  
- `scikit-learn` 1.3.2  
- `statsmodels` 0.14.1  
- `numba` 0.58.1  
- `networkx` 3.1  
- `h5py` 3.11.0  

### Single-cell / dimensionality reduction

- `anndata` 0.9.2  
- `scanpy` 1.9.8  
- `umap-learn` 0.5.7  
- `pynndescent` 0.5.13 fileciteturn0file0

### Optimal transport and interpretability

- `pot` 0.9.5 (Python Optimal Transport, used for GW and prior losses) fileciteturn1file10  
- `captum` 0.7.0  

### Visualization, logging and utilities

- `matplotlib` 3.7.5  
- `seaborn` 0.13.2  
- `tensorboardx` 2.6.2.2 fileciteturn1file1  
- `tqdm` 4.67.1  
- `pyyaml` 5.4.1  
- `jinja2` 3.1.6  

All remaining packages in the environment (`session-info`, `natsort`, `pillow`, etc.) are standard dependencies and do not need special configuration.

### Minimal installation example

After creating and activating the Conda environment, the core dependencies can be installed with:

```bash
pip install   torch torchvision torchaudio   anndata scanpy umap-learn pynndescent   pot captum tensorboardx   numpy pandas scipy scikit-learn statsmodels   matplotlib seaborn numba networkx h5py pyyaml jinja2 tqdm
```

(You can also export and share the exact environment with `conda list --export > environment.txt` for full reproducibility.)

---

## 2. Code Structure

The main Python modules in this repository are:

- **`data_loader.py`**
  - Defines `CrossModalDataset`, which loads and standardizes:
    - morphology data from a CSV (e.g. `gw_dist.csv`),  
    - gene expression data from a CSV (e.g. `exon_data_top2000.csv`),  
    - RNA family labels (e.g. `rna_family_matched.csv`),  
    - morphology cluster labels (e.g. `cluster_label_morpho.csv`),  
    - gene expression cluster labels (e.g. `cluster_label_GEX.csv`),  
    - a prior morphology–transcriptomics correlation matrix (e.g. `Corr_matrix.csv`). fileciteturn1file4  
  - Standardizes each modality with `StandardScaler` and truncates all arrays to the common number of samples.  
  - Exposes:
    - `__getitem__` returning a dictionary with keys:  
      `morpho_data`, `gex_data`, `morpho_cluster`, `gex_cluster`, `index`, `rna_family`. fileciteturn1file2  
    - `get_full_data(device)` – returns full morphology and gene expression tensors. fileciteturn1file2  
    - `get_prior_matrix(device)` – returns the prior correlation matrix tensor. fileciteturn1file2  
    - `get_rna_family_labels()` – returns RNA family labels. fileciteturn1file2  
  - Also provides `create_data_loader(dataset, batch_size, shuffle, num_workers)` to construct a `DataLoader`. fileciteturn1file2  

- **`utils.py`**
  - General training utilities:  
    - `get_cross_modal_data_loader` – creates a `CrossModalDataset` and corresponding `DataLoader`. fileciteturn1file0  
    - `get_config` – loads a YAML configuration file. fileciteturn1file13  
    - `prepare_sub_folder` – creates `images/` and `checkpoints/` subdirectories inside an output directory. fileciteturn1file13  
    - `write_loss` – logs scalar losses/gradients to TensorBoard. fileciteturn1file13  
    - `get_model_list`, `get_scheduler`, `weights_init`. fileciteturn1file13  
  - Evaluation and visualization helpers:
    - Latent space visualizations (PCA / UMAP) and `save_plots(...)` for saving modality-aligned embeddings. fileciteturn1file15  
    - KNN-based matching and label-accuracy evaluation via `write_knn(...)`. fileciteturn0file4  

- **`networks.py`**
  - Defines:
    - `Discriminator` – a fully connected discriminator operating on the shared latent space, supporting LSGAN / NSGAN / WGAN losses and optional gradient penalty. fileciteturn0file1  
    - `VAEGen_MORE_LAYERS` (imported as `VAEGen`) – VAE encoder/decoder with layer depth adapted to input dimensionality for each modality. fileciteturn0file1  
    - (Optionally) a `Classifier` used by the trainer. fileciteturn1file10  

- **`trainer.py`**
  - Defines the `Trainer` class, which:
    - Builds two generators (`gen_a` and `gen_b`) and a latent-space discriminator (`dis_latent`). fileciteturn1file10  
    - Sets up Adam optimizers and learning-rate schedulers. fileciteturn1file10  
    - Implements a **configurable loss schedule** and **training phases** (`phase_1`, `phase_2`, `phase_3`, …) via `loss_schedule` and `phase_durations` in the config. fileciteturn1file10  
    - Implements generator updates (`gen_update`) and discriminator updates (`dis_update`) with:
      - Reconstruction loss for both modalities,  
      - KL loss for both modalities (variational term),  
      - Optional GAN loss in latent space,  
      - Optional GW loss between latent spaces,  
      - Optional prior loss using the precomputed correlation matrix and cluster labels. fileciteturn1file7turn1file9turn1file10  
    - Tracks loss histories and scalar summaries for logging. fileciteturn1file10  

- **`train.py`**
  - Main entry point for training.  
  - Parses command line arguments:
    - `--config` (required): path to a YAML config. fileciteturn1file1  
    - `--output_path`: base directory for logs and outputs (default: current directory). fileciteturn1file1  
    - `--resume`: flag to resume from the latest checkpoint in `output_path/outputs/<config_name>/checkpoints`. fileciteturn1file1  
  - Loads configuration, initializes data loader and dataset via `get_cross_modal_data_loader`, and constructs a `Trainer`. fileciteturn1file1  
  - Creates TensorBoard log directory and output directories:  
    - Logs: `<output_path>/logs/<config_basename>/`  
    - Outputs: `<output_path>/outputs/<config_basename>/images/` and `.../checkpoints/`. fileciteturn1file1  
  - Runs the main training loop with:
    - Phase transitions and printing of active losses. fileciteturn1file6  
    - Periodic discriminator analysis, logging, and latent space statistics. fileciteturn1file8turn1file12  
    - Saving checkpoints, plots, KNN results, and loss curves at configured intervals and at the end of training. fileciteturn1file12  

- **Slurm script (e.g., `train.slurm`)**
  - A convenience script for submitting training jobs to a Slurm cluster (not required for local runs).

- **Config file(s) (e.g., `attempt_1.yaml`)**
  - YAML files containing all hyperparameters and scheduling options used by `Trainer`. fileciteturn0file3  

---

## 3. Data Requirements

All data are loaded through `CrossModalDataset` in `data_loader.py`. By default, the dataset expects the following CSV files (column indexing is handled inside the code):

- **Morphology features** (modality A)  
  - Example: `gw_dist.csv`  
  - Used as `morpho_data` (all columns except the first are treated as numeric features). fileciteturn1file4  

- **Gene expression features** (modality B)  
  - Example: `exon_data_top2000.csv`  
  - Used as `gex_data` (again, all columns except the first are treated as numeric features). fileciteturn1file4  

- **RNA family labels**  
  - Example: `rna_family_matched.csv`  
  - Used for evaluation and plotting; labels are read from the first non-index column. fileciteturn1file4turn1file0  

- **Morphology cluster labels**  
  - Example: `cluster_label_morpho.csv`  
  - Used to derive integer cluster IDs for modality A. fileciteturn1file4  

- **Gene expression cluster labels**  
  - Example: `cluster_label_GEX.csv`  
  - Used to derive integer cluster IDs for modality B. fileciteturn1file4  

- **Prior correlation matrix**  
  - Example: `Corr_matrix.csv`  
  - Loaded as a 2D correlation matrix relating gene expression clusters to morphology clusters, used to define a prior alignment term. fileciteturn1file4turn1file3  

> **Note:** In the original code these files are referenced via absolute paths. For portability, you can change the default arguments of `CrossModalDataset` in `data_loader.py` to point to your local data files (e.g. relative paths inside a `data/` folder). fileciteturn1file4

---

## 4. Running Training

### 4.1. Prepare a config file

Create a YAML config file (for example `configs/attempt_1.yaml`) specifying:

- Global training parameters:
  - `max_iter` – total training iterations. fileciteturn1file1  
  - `batch_size` – batch size used in `get_cross_modal_data_loader`. fileciteturn1file1  
  - `lr`, `beta1`, `beta2`, `weight_decay` – optimizer settings. fileciteturn1file10  
  - `init` – weight initialization type (`gaussian`, `xavier`, `kaiming`, etc.). fileciteturn1file13  
  - Learning rate schedule: `lr_policy`, `step_size`, `gamma` (if step schedule is used). fileciteturn1file13  

- Network dimensions:
  - `input_dim_a` – morphology feature dimension (number of columns used from morphology CSV). fileciteturn1file10turn1file4  
  - `input_dim_b` – gene expression feature dimension. fileciteturn1file10turn1file4  
  - `gen` block with keys like `dim` and `latent` controlling encoder/decoder widths and latent dimensionality. fileciteturn1file10turn0file1  
  - `dis` block controlling `gan_type`, `dim`, `norm`, etc. for the discriminator. fileciteturn0file1turn1file10  

- Loss scheduling and phases:
  - `loss_schedule` – a mapping from loss names to starting phases, e.g.  
    ```yaml
    loss_schedule:
      kl_loss: 1
      recon_loss: 1
      gan_loss: 2
      gw_loss: 3
      prior_loss: 1
    ``` fileciteturn1file10  
  - `phase_durations` – how many iterations each training phase lasts, e.g.  
    ```yaml
    phase_durations:
      phase_1: 200
      phase_2: 400
      phase_3: 400
    ``` fileciteturn1file10turn1file7  

- Loss weights (examples):
  - `recon_x_w`, `kl_w`, `gan_w`, `gw_w`, `lambda_p`. fileciteturn1file9turn1file14  
  - Optional prior-specific parameters such as `prior_temperature`, `prior_loss_warmup`. fileciteturn0file3turn1file3  

- Visualization settings (optional):
  - `visualization` block with keys like `method` (`pca` or `umap`), `umap_n_neighbors`, `umap_min_dist`, `umap_metric`. fileciteturn1file15  

You can use your existing `attempt_1.yaml` as a template and adapt values as needed.

### 4.2. Launch training

From the repository root:

```bash
python train.py --config configs/attempt_1.yaml --output_path ./results
```

- `--config` points to your YAML configuration.  
- `--output_path` is a base directory; the code will create:
  - `./results/logs/<config_name>/` – TensorBoard logs. fileciteturn1file1  
  - `./results/outputs/<config_name>/images/` – saved plots and visualizations. fileciteturn1file1turn1file15  
  - `./results/outputs/<config_name>/checkpoints/` – model checkpoints. fileciteturn1file1turn1file13  

To resume training from the latest checkpoint in the same output directory:

```bash
python train.py --config configs/attempt_1.yaml --output_path ./results --resume
``` fileciteturn1file1turn1file12  

During training, the script prints:

- Current phase and active losses,
- Loss breakdown (KL, reconstruction, GAN, GW, prior),
- Latent space statistics (means, standard deviations, cross-modal correlations, cosine similarities). fileciteturn1file8turn1file9turn1file12  

At the end of training it saves:

- Final model checkpoints,  
- Final plots (PCA/UMAP),  
- Loss curves and KNN-based evaluation results. fileciteturn1file12turn0file4  

---

## 5. Outputs and Logging

Without creating a separate numbered “Evaluation & Visualization” section, the main outputs of the training pipeline are:

- **Model checkpoints**  
  - Saved in `outputs/<config_name>/checkpoints/` and used for resume or downstream analysis. fileciteturn1file12turn1file13  

- **Latent space plots**  
  - Saved in `outputs/<config_name>/images/` by `save_plots(...)`.  
  - Depending on configuration, show PCA or UMAP projections of the morphology and gene expression latent spaces, colored by RNA family labels (if available). fileciteturn1file15turn0file4  

- **KNN-based matching metrics**  
  - Written to `knn_accuracy.txt` inside the `images/` directory.  
  - Includes both position-based KNN accuracy and label-matching accuracy for RNA family labels across modalities. fileciteturn0file4turn1file12  

- **Loss curves**  
  - `save_loss_curves(...)` and `save_weighted_loss_curves(...)` produce figures summarizing the evolution of individual losses and their weighted combinations over training. fileciteturn0file4turn1file12  

- **TensorBoard logs**  
  - Scalars written via `write_loss(...)` can be visualized with TensorBoard:  
    ```bash
    tensorboard --logdir ./results/logs
    ``` fileciteturn1file13turn1file1  

---

## 6. Reproducibility

To share the exact environment used for a given experiment, you can export:

```bash
# Conda-style export
conda list --export > environment.txt

# Pip-style export
pip freeze > requirements.txt
```

Include these files in the repository if you want the paper reviewers or other users to reproduce the results exactly.

---

## 7. Citation

Please add your **final manuscript citation** here once it is available, for example:

```text
[To be added]  Author list, title, journal/conference, year.
```

You can then reference this GitHub repository in your paper and in this README.

---

## 8. License

If you plan to make the repository public, add a `LICENSE` file (for example, MIT or BSD-3-Clause) and mention it here, for example:

```text
This code is released under the MIT License. See LICENSE for details.
```

(Replace with your actual license choice.)

---

## 9. Contact

For questions related to this codebase, please refer to the contact information you provide in the manuscript or in the GitHub repository (e.g. maintainer’s email or GitHub handle).
