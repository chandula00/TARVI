# TARVI

**Transcription-Factor Aided RNA Velocity Inference** — a deep generative (VAE)
extension of VeloVI with cell-specific TF-regulated transcription rates
(ChEA/ENCODE), a supervised residual latent-time head, a velocity–time
consistency loss, and velocity–pseudotime blending for calibrated cell-fate
trajectories from scRNA-seq.

---

<p align="center">
  <img src="docs/assets/architecture.png" alt="TARVI architecture" width="100%">
</p>

<p align="center"><em>TARVI architecture — the VeloVI VAE backbone augmented with a
TF-regulated transcription rate, a supervised residual latent-time head, ODE-aligned
auxiliary losses, and a velocity–pseudotime blend.</em></p>

## Overview

RNA velocity infers cellular dynamics from the ratio of unspliced to spliced
mRNA, but contemporary estimators share three structural limitations that TARVI
targets directly:

1. **Static transcription rates.** The per-gene rate `α_g` is a constant that
   ignores cell-state-dependent regulation. TARVI computes a **cell-specific
   `α_i`** from a masked, learnable transcription-factor → gene weight matrix
   built from ENCODE ChIP-seq and ChEA TF–target priors.
2. **Weakly-supervised latent time.** TARVI adds a **supervised residual
   time head** (trained against a stop-gradient ODE-time signal) with a
   graph-Laplacian temporal-smoothness regulariser and a **velocity–time
   consistency loss** that penalises velocities pointing from later to earlier
   cells.
3. **Local/global trajectory mismatch.** A post-hoc **velocity–pseudotime (VPT)
   blending** step fuses the learned time head with diffusion pseudotime on the
   velocity transition graph for a globally grounded temporal coordinate.

The model is built on the VeloVI four-state (induction / induction-steady /
repression / repression-steady) generative backbone.

## Repository structure

```
TARVI/
├── tarvi/                     # core model package
│   ├── _model.py             # TARVI model class (scvi-tools BaseModelClass)
│   ├── _module.py            # TARVIVAE generative module + decoder
│   ├── _tf_module.py         # TF-regulated transcription rate + build_tf_mask
│   ├── _gnn.py               # graph refinement utilities
│   ├── _distributions.py     # custom distributions
│   ├── _utils.py             # preprocessing (preprocess_data)
│   └── _constants.py
├── evaluation/                # benchmarking metrics (ICCoH, CBDir, pseudotime, …)
│   └── metrics.py
├── scripts/                   # training / benchmark / ablation / figures
│   ├── train.py
│   ├── run_benchmark.py
│   ├── run_ablation.py
│   └── plot_radar.py
├── paper/                     # compiled manuscript
│   └── TARVI_paper.pdf
├── environment.yml
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate tarvi
pip install -e .
```

The environment pins PyTorch 2.10 with CUDA 12.8; adjust the `torch` wheel index
in `environment.yml` for your CUDA version.

## Data

- **Built-in datasets** (`pancreas`, `forebrain`, `bonemarrow`, `dentategyrus`)
  download automatically via `scvelo.datasets`.
- **Additional benchmark datasets** (`chromaffin`, `sceu_organoid`, …) should be
  placed under `./data/` (see the dataset table in the paper for sources).
- **TF–target databases.** The TF-regulation module needs the ENCODE and ChEA
  TF–target files (the same Enrichr / ChEA 2016 + ENCODE resources used by
  TFvelo). Place them under `./data/TFvelo/` (with `ENCODE/` and `ChEA/`
  subfolders) and pass the directory via `--tf_data_dir` (or the `tf_data_dir=`
  argument).

> `data/` is git-ignored; datasets are public and downloaded/placed locally.

## Quick start (Python API)

```python
import scvelo as scv
from tarvi import TARVI, preprocess_data

adata = scv.datasets.pancreas()
adata = preprocess_data(adata)                  # QC, moments, MinMax scaling, r² gene filter

TARVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
model = TARVI(
    adata,
    n_latent=10,
    use_tf_regulation=True,                     # cell-specific TF-regulated α_i
    tf_data_dir="./data/TFvelo",                # ENCODE + ChEA TF–target priors
)
model.train(max_epochs=500, accelerator="gpu", devices=[0])

adata.layers["velocity"] = model.get_velocity(n_samples=25, smooth=True).values
adata.obs["latent_time"] = model.get_latent_time(n_samples=25).mean(axis=1)
adata.obsm["X_tarvi"]    = model.get_latent_representation()
```

## Reproducing the paper

```bash
# 1) Train a single TARVI model  (modes: baseline | tf | tf_nb | full)
python scripts/train.py --dataset pancreas --mode full \
    --tf_data_dir ./data/TFvelo --log outputs/pancreas_full.log

# 2) Full benchmark on one dataset (TARVI + all baselines, all metrics)
python scripts/run_benchmark.py --dataset pancreas \
    --tf_data_dir ./data/TFvelo --log outputs/bench_pancreas.log

# 3) Component ablation across the five datasets (Table in the supplement)
python scripts/run_ablation.py \
    --datasets pancreas bonemarrow forebrain chromaffin sceu_organoid \
    --tf_data_dir ./data/TFvelo

# 4) Cross-dataset radar figure
python scripts/plot_radar.py
```

Use `--help` on any script for the full option list.

## Results (cross-dataset means, 5 datasets)

| Method | CBDir ↑ | VelConf ↑ | PT-Spear ↑ | PT-DistCorr ↑ | PT-Cons ↑ | Gene-R²ₛₚₗ ↑ |
|--------|:------:|:--------:|:---------:|:------------:|:--------:|:-----------:|
| **TARVI**  | **0.933** | **0.927** | **0.747** | **0.771** | **0.986** | **0.477** |
| VeloVI | 0.931 | 0.920 | 0.331 | 0.341 | 0.904 | 0.443 |

TARVI matches the strongest baseline on cross-boundary direction accuracy while
substantially improving pseudotime calibration (+126% Spearman over VeloVI). See
`paper/` and `paper/supplementary_ablation.tex` for the full per-dataset tables,
all eight metrics, four baselines, and the component ablation.

## Paper & supplementary

The paper PDF will be added to [`paper/`](paper/) **after publication**. A supplementary
document covering the full component ablation, the background theory it relies on, and
reproducibility/scope notes is already available in [`supplementary/`](supplementary/)
(`supplementary_material.pdf`, with LaTeX source).

## Citation

If you use TARVI, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{adhikari2026tarvi,
  title     = {TARVI: Transcription-Factor Aided RNA Velocity Inference with
               Supervised Latent Time and Velocity-Pseudotime Blending},
  author    = {Adhikari, Chandula and Dassanayake, Sandeep and Herath, Damayanthi},
  year      = {2026}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Acknowledgements

TARVI builds on the VeloVI generative backbone (`scvi-tools`) and the
Scanpy/scVelo single-cell stack, and uses ENCODE and ChEA TF–target annotations
for the transcription-factor regulation module.
