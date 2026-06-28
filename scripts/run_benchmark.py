"""TARVI on multiple scRNA-seq RNA velocity datasets.

Dentate Gyrus  (~2930 cells) — scvelo.datasets.dentategyrus()
    embedding  : X_umap  (pre-computed)
    basis      : umap
    cluster key: clusters

Bone Marrow / Hematopoiesis (~5780 cells) — scvelo.datasets.bonemarrow()
    embedding  : X_tsne  (pre-computed)
    basis      : tsne
    cluster key: clusters
    ground truth: palantir_pseudotime

Human Forebrain (~1720 cells) — loom file
    embedding  : X_umap  (computed from PCA)
    basis      : umap
    cluster key: Clusters

Mouse Gastrulation Erythroid (~9815 cells) — scvelo.datasets.gastrulation_erythroid()
    embedding  : X_umap  (pre-computed)
    basis      : umap
    cluster key: celltype

Direct Reprogramming — cellrank.datasets.reprogramming_morris()
    embedding  : X_umap  (computed from PCA)
    basis      : umap
    cluster key: clusters
    ground truth (day labels): day

PBMC 68k (Negative Control) — scvelo.datasets.pbmc68k()
    embedding  : X_tsne  (pre-computed)
    basis      : tsne
    cluster key: celltype

Zebrafish Heart Regeneration — cellrank.datasets.zebrafish()
    embedding  : X_umap  (computed from PCA)
    basis      : umap
    cluster key: clusters

scEU-seq Organoid (Metabolic) — dynamo.sample_data.scEU_seq_organoid()
    embedding  : X_umap  (computed from PCA)
    basis      : umap
    cluster key: cell_type

Usage:
    conda activate tarvi
    # Dentate Gyrus
    python run_new_datasets.py --dataset dentate_gyrus --gpu 0 \\
        --log outputs/dentate_gyrus/run.log

    # Bone Marrow
    python run_new_datasets.py --dataset bonemarrow --gpu 0 \\
        --log outputs/bonemarrow/run.log

    # Mouse Gastrulation
    python run_new_datasets.py --dataset gastrulation --gpu 0 \\
        --log outputs/gastrulation/run.log

    # Direct Reprogramming
    python run_new_datasets.py --dataset reprogramming --gpu 0 \\
        --log outputs/reprogramming/run.log
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Set before any pyrovelocity/jax import
os.environ.setdefault("PYROVELOCITY_TESTING_FLAG", "False")
os.environ.setdefault("PYROVELOCITY_LOG_LEVEL", "ERROR")
# UniTVelo requires Keras 2 (tf.keras.optimizers.legacy); set before TF loads
os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tarvi import TARVI, preprocess_data
from evaluation.metrics import compute_all_metrics

log = logging.getLogger("tarvi.newdata")

# ─── Dataset configs ──────────────────────────────────────────────────────────

_FB_LOOM = ("./data"
            "/Human_Forebrain/data/hgForebrainGlut.loom")

_CHROM_DIR = ("./data"
              "/Chromaffin differentiation")

_CHROM_HEX_TO_CELLTYPE = {
    "#FF0000": "Chromaffin",
    "#0066FF": "Sympathoblast",
    "#00FF66": "SCP",
    "#CCFF00": "Bridge",
    "#CC00FF": "NCC",
}

def _hex_to_celltype(hex_color):
    short = hex_color[:7].upper()
    for key, name in _CHROM_HEX_TO_CELLTYPE.items():
        if key.upper() == short:
            return name
    return f"Unknown_{short}"

DATASET_CONFIG = {
    "dentate_gyrus": dict(
        loader="scv.datasets.dentategyrus",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=False,
        cluster_key="clusters",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Dentate Gyrus Neurogenesis",
        output_dir="outputs/dentate_gyrus",
        dot_size_stream=60,   # scvelo stream size=
        dot_size_scatter=20,  # matplotlib scatter s=
    ),
    "bonemarrow": dict(
        loader="scv.datasets.bonemarrow",
        loom_path=None,
        embedding_key="X_tsne",
        basis="tsne",
        compute_embedding=False,
        cluster_key="clusters",
        gt_pseudotime_key="palantir_pseudotime",
        gt_time_key=None,
        title="Hematopoiesis (Bone Marrow)",
        output_dir="outputs/bonemarrow",
        dot_size_stream=60,
        dot_size_scatter=20,
    ),
    "forebrain": dict(
        loader=None,
        loom_path=_FB_LOOM,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=True,
        umap_n_neighbors=30,      # from original paper evaluation
        umap_min_dist=0.3,
        loom_tsne_cols=None,
        cluster_key="Clusters",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Human Forebrain (Glutamatergic)",
        output_dir="outputs/forebrain",
        dot_size_stream=100,
        dot_size_scatter=40,
    ),
    "gastrulation": dict(
        loader="scv.datasets.gastrulation_erythroid",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=False,
        cluster_key="celltype",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Mouse Gastrulation (Erythroid)",
        output_dir="outputs/gastrulation",
        dot_size_stream=40,
        dot_size_scatter=15,
    ),
    "reprogramming": dict(
        loader="cellrank.datasets.reprogramming_morris",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=True,
        cluster_key="clusters",
        gt_pseudotime_key=None,
        gt_time_key="day",
        title="Direct Reprogramming",
        output_dir="outputs/reprogramming",
        dot_size_stream=60,
        dot_size_scatter=20,
    ),
    "pbmc68k": dict(
        loader="scv.datasets.pbmc68k",
        loom_path=None,
        embedding_key="X_tsne",
        basis="tsne",
        compute_embedding=False,
        cluster_key="celltype",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="PBMC 68k (Negative Control)",
        output_dir="outputs/pbmc68k",
        dot_size_stream=30,
        dot_size_scatter=10,
    ),
    "zebrafish": dict(
        loader="cellrank.datasets.zebrafish",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=True,
        cluster_key="clusters",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Zebrafish Heart Regeneration",
        output_dir="outputs/zebrafish",
        dot_size_stream=60,
        dot_size_scatter=20,
    ),
    "sceu_organoid": dict(
        loader="sceu_standalone",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=False,
        cluster_key="clusters",
        gt_pseudotime_key="monocle_pseudotime",
        gt_time_key="time_numeric",
        title="scEU-seq Organoid (Metabolic)",
        output_dir="outputs/sceu_organoid",
        dot_size_stream=60,
        dot_size_scatter=20,
    ),
    "chromaffin": dict(
        loader="loom",
        loom_path=f"{_CHROM_DIR}/data/velocyto/onefilepercell_A1_unique_and_others_J2CH1.loom",
        embedding_key="X_tsne",
        basis="tsne",
        compute_embedding=False,
        cluster_key="clusters",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Chromaffin Differentiation",
        output_dir="outputs/chromaffin",
        dot_size_stream=80,
        dot_size_scatter=30,
        # Chromaffin-specific extras (read by load_dataset)
        chrom_tsne_path=f"{_CHROM_DIR}/data/embedding.csv",
        chrom_colors_path=f"{_CHROM_DIR}/data/cell_colors.csv",
    ),
    "pancreas": dict(
        loader="scv.datasets.pancreas",
        loom_path=None,
        embedding_key="X_umap",
        basis="umap",
        compute_embedding=False,
        cluster_key="clusters",
        gt_pseudotime_key=None,
        gt_time_key=None,
        title="Pancreatic Endocrinogenesis",
        output_dir="outputs/pancreas",
        dot_size_stream=60,
        dot_size_scatter=20,
    ),
}

TARVI_MODE_CONFIG = {
    "baseline": dict(use_tf=False, ode_residual_weight=1.0, vpt_blend=0.0, use_vpt=False),
    "tf":       dict(use_tf=True,  ode_residual_weight=1.0, vpt_blend=0.0, use_vpt=False),
    "full":     dict(use_tf=True,  ode_residual_weight=1.0, vpt_blend=0.7, use_vpt=True),
}

CLUSTER_PALETTE = [
    "#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A",
    "#B15928", "#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F",
    "#CAB2D6", "#FFFF99", "#1B9E77", "#D95F02", "#7570B3",
]


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_logging(log_file):
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(dataset_name, args):
    cfg = DATASET_CONFIG[dataset_name]
    log.info(f"Loading {dataset_name} ...")

    # ── Load raw data ─────────────────────────────────────────────────────────
    if dataset_name == "chromaffin":
        loom_path = cfg["loom_path"]
        adata = sc.read_loom(loom_path, sparse=True)
        # Clean obs names: 'file:barcode_unique.bam' → 'barcode'
        adata.obs_names = [
            name.split(":")[-1].replace("_unique.bam", "")
            for name in adata.obs_names
        ]
        log.info(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

        tsne_df   = pd.read_csv(cfg["chrom_tsne_path"]).set_index("cell_id")
        colors_df = pd.read_csv(cfg["chrom_colors_path"]).set_index("cell_id")
        valid_cells = (
            adata.obs_names.intersection(tsne_df.index).intersection(colors_df.index)
        )
        log.info(f"  Valid cells: {len(valid_cells)}")
        adata = adata[valid_cells].copy()
        adata.obsm["X_tsne"] = tsne_df.loc[valid_cells, ["V1", "V2"]].values
        raw_colors = colors_df.loc[valid_cells, "cell_colors"].values
        adata.obs["clusters"] = pd.Categorical(
            [_hex_to_celltype(c) for c in raw_colors]
        )
        adata.obs["cell_colors"] = raw_colors
    elif cfg["loom_path"]:
        adata = sc.read_loom(cfg["loom_path"], sparse=True)
        # Standardise spliced/unspliced layer names (loom often has 'matrix')
        if "spliced" not in adata.layers and "matrix" in adata.layers:
            adata.layers["spliced"] = adata.layers["matrix"]
        # Extract pre-computed t-SNE from obs columns into obsm BEFORE any filtering
        tsne_cols = cfg.get("loom_tsne_cols")
        if tsne_cols and tsne_cols[0] in adata.obs.columns:
            adata.obsm["X_tsne"] = adata.obs[list(tsne_cols)].values.astype(float)
            log.info(f"  Loaded pre-computed t-SNE from loom obs columns {tsne_cols}")
    elif dataset_name == "dentate_gyrus":
        adata = scv.datasets.dentategyrus()
    elif dataset_name == "bonemarrow":
        adata = scv.datasets.bonemarrow()
    elif dataset_name == "gastrulation":
        adata = scv.datasets.gastrulation_erythroid()
    elif dataset_name == "reprogramming":
        try:
            import cellrank
            adata = cellrank.datasets.reprogramming_morris()
        except ImportError:
            raise RuntimeError(
                "cellrank is not installed. Cannot load reprogramming dataset. "
                "Install with: pip install cellrank"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load reprogramming dataset: {e}")
    elif dataset_name == "pbmc68k":
        adata = scv.datasets.pbmc68k()
    elif dataset_name == "zebrafish":
        try:
            import cellrank
            adata = cellrank.datasets.zebrafish()
        except ImportError:
            raise RuntimeError(
                "cellrank is not installed. Cannot load zebrafish dataset. "
                "Install with: pip install cellrank"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load zebrafish dataset: {e}")
    elif dataset_name == "sceu_organoid":
        # Loaded from preprocessed h5ad saved by run_sceu_standalone.py (sceu_env)
        preproc = os.path.join(cfg["output_dir"], "preprocessed.h5ad")
        if not os.path.isfile(preproc):
            raise RuntimeError(
                f"sceu_organoid preprocessed h5ad not found: {preproc}\n"
                "Run: conda run -n sceu_env python run_sceu_standalone.py "
                f"--output {preproc}"
            )
        adata = sc.read_h5ad(preproc)
    elif dataset_name == "pancreas":
        adata = scv.datasets.pancreas()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log.info(f"  Raw: {adata.n_obs} cells, {adata.n_vars} genes")
    log.info(f"  Layers: {list(adata.layers.keys())}")
    log.info(f"  Obsm: {list(adata.obsm.keys())}")
    log.info(f"  Obs cols: {list(adata.obs.columns)}")

    cluster_key = cfg["cluster_key"]
    # For sceu_organoid, try fallback cluster keys if primary not found
    if cluster_key not in adata.obs and dataset_name == "sceu_organoid":
        for fallback in ["som_cluster_id", "clusters", "cell_type", "exp_type"]:
            if fallback in adata.obs:
                log.info(f"  Primary cluster key '{cluster_key}' not found, using fallback '{fallback}'")
                cluster_key = fallback
                cfg = dict(cfg)
                cfg["cluster_key"] = cluster_key
                break
        else:
            raise RuntimeError(
                f"No valid cluster key found in adata.obs for sceu_organoid. "
                f"Available: {list(adata.obs.columns)}"
            )
    elif cluster_key not in adata.obs:
        raise RuntimeError(f"Cluster key '{cluster_key}' not found in adata.obs")

    # Ensure cluster column is Categorical
    if not hasattr(adata.obs[cluster_key], "cat"):
        unique_types = sorted(adata.obs[cluster_key].dropna().unique().tolist(),
                              key=str)
        adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key],
                                                 categories=unique_types)
    unique_types = adata.obs[cluster_key].cat.categories.tolist()
    adata.uns[f"{cluster_key}_colors"] = CLUSTER_PALETTE[:len(unique_types)]

    log.info(f"  Cell types ({cluster_key}): {dict(adata.obs[cluster_key].value_counts())}")
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata.obs:
        log.info(f"  Ground truth pseudotime: '{cfg['gt_pseudotime_key']}' present")
    if cfg.get("gt_time_key") and cfg["gt_time_key"] in adata.obs:
        log.info(f"  Ground truth time key: '{cfg['gt_time_key']}' present")

    # ── Preprocessing ────────────────────────────────────────────────────────
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    log.info(f"  After QC: {adata.n_obs} cells, {adata.n_vars} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
        n_top_genes=args.n_top_genes,
    )
    adata = adata[:, adata.var.highly_variable].copy()
    log.info(f"  After HVG: {adata.n_obs} cells, {adata.n_vars} genes")

    # ── Compute embedding if not pre-computed ─────────────────────────────────
    emb_key = cfg["embedding_key"]
    basis    = cfg["basis"]

    if cfg["compute_embedding"]:
        log.info(f"  Computing PCA → {basis} embedding...")
        sc.tl.pca(adata, svd_solver="arpack", n_comps=30)
        if basis == "umap":
            umap_nn  = cfg.get("umap_n_neighbors", args.n_neighbors)
            umap_md  = cfg.get("umap_min_dist", 0.5)
            sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=umap_nn)
            sc.tl.umap(adata, min_dist=umap_md)
            log.info(f"  UMAP params: n_neighbors={umap_nn}, min_dist={umap_md}")
        elif basis == "tsne":
            sc.tl.tsne(adata, use_rep="X_pca", n_jobs=4)
        log.info(f"  Embedding computed: {emb_key} shape {adata.obsm[emb_key].shape}")
    else:
        if emb_key not in adata.obsm:
            raise RuntimeError(
                f"Pre-computed embedding '{emb_key}' not found. "
                f"Available: {list(adata.obsm.keys())}"
            )

    # ── Neighbors + moments ───────────────────────────────────────────────────
    # If compute_embedding already ran neighbors (umap/tsne), don't redo it
    if not cfg["compute_embedding"] or basis == "tsne":
        # pre-computed embedding: compute neighbors on it now
        sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=args.n_neighbors)
        scv.pp.moments(adata, n_pcs=None, n_neighbors=args.n_neighbors,
                       use_rep=emb_key)
    else:
        # neighbors already computed during compute_embedding (umap path)
        scv.pp.moments(adata, n_pcs=None, n_neighbors=args.n_neighbors,
                       use_rep="X_pca")

    log.info(f"  Moments computed. Layers: {list(adata.layers.keys())}")
    return adata, cfg


# ─── Model runners ────────────────────────────────────────────────────────────

def run_velocyto(adata_raw, cfg, args):
    """Velocyto steady-state model (La Manno 2018).

    Implemented via scVelo's mode='deterministic', which is the same algorithm.
    No pseudotime is available from this model; velocity_pseudotime is used instead.
    """
    log.info("\n" + "=" * 70)
    log.info("  MODEL: velocyto (steady-state)")
    log.info("=" * 70)

    adata = adata_raw.copy()
    start = time.time()
    scv.tl.velocity(adata, mode="deterministic")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)
    adata.obs["latent_time"] = adata.obs["velocity_pseudotime"]
    train_time = time.time() - start
    log.info(f"  Time: {train_time:.1f}s")

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    adata.obsm["X_latent"] = adata.obsm["X_pca"]

    gt_time_key = cfg.get("gt_time_key")
    metrics = compute_all_metrics(
        adata,
        velocity_key="velocity",
        latent_time_key="latent_time",
        latent_key="X_latent",
        cluster_key=cfg["cluster_key"],
        umap_key=cfg["embedding_key"],
        gt_time_key=gt_time_key,
    )
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
        from scipy.stats import spearmanr
        gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
        pred = adata.obs["latent_time"]
        corr, _ = spearmanr(pred, gt)
        metrics["gt_pseudotime_spearman"] = float(abs(corr))
        log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["model"] = "velocyto"
    return adata, metrics


def run_scvelo_dynamical(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: scVelo Dynamical")
    log.info("=" * 70)

    adata = adata_raw.copy()
    start = time.time()
    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.tl.latent_time(adata)
    train_time = time.time() - start
    log.info(f"  Time: {train_time:.1f}s")

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    adata.obsm["X_latent"] = adata.obsm["X_pca"]

    gt_time_key = cfg.get("gt_time_key")
    metrics = compute_all_metrics(
        adata,
        velocity_key="velocity",
        latent_time_key="latent_time",
        latent_key="X_latent",
        cluster_key=cfg["cluster_key"],
        umap_key=cfg["embedding_key"],
        gt_time_key=gt_time_key,
    )
    # Ground-truth pseudotime correlation if available
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
        from scipy.stats import spearmanr
        gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
        pred = adata.obs["latent_time"]
        corr, _ = spearmanr(pred, gt)
        metrics["gt_pseudotime_spearman"] = float(abs(corr))
        log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["model"] = "scVelo_dyn"
    return adata, metrics


def run_velovi(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: VeloVI")
    log.info("=" * 70)

    adata = adata_raw.copy()
    adata = preprocess_data(adata)
    log.info(f"  Shape after preprocess: {adata.shape}")

    try:
        from velovi import VELOVI
        import velovi._module as _velovi_mod
    except ImportError:
        log.error("  velovi not installed. Skipping.")
        return None, None

    # CUDA Dirichlet KL patch
    from tarvi._module import _DirichletKLFunction

    def _patched_loss(self, tensors, inference_outputs, generative_outputs,
                      kl_weight=1.0, n_obs=1.0):
        from torch.distributions import Normal
        from torch.distributions.kl import kl_divergence as kl
        from scvi.module.base import LossOutput

        spliced   = tensors["X"]
        unspliced = tensors["U"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_pi       = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]
        end_penalty = generative_outputs["end_penalty"]

        kl_divergence_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)
        ).sum(dim=1)
        reconst_loss_s = -generative_outputs["mixture_dist_s"].log_prob(spliced)
        reconst_loss_u = -generative_outputs["mixture_dist_u"].log_prob(unspliced)
        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        alpha_p = torch.clamp(px_pi_alpha, min=0.01, max=1e6)
        alpha_q = torch.clamp(
            self.dirichlet_concentration * torch.ones_like(px_pi), min=0.01, max=1e6,
        )
        kl_pi = _DirichletKLFunction.apply(alpha_p, alpha_q).sum(dim=-1)
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * kl_divergence_z + kl_pi
        local_loss = torch.mean(reconst_loss + weighted_kl_local)
        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

    _velovi_mod.VELOVAE.loss = _patched_loss

    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    model = VELOVI(adata, n_hidden=256, n_latent=10)

    start = time.time()
    model.train(
        max_epochs=args.max_epochs, lr=1e-2,
        batch_size=min(256, max(32, adata.n_obs // 4)),
        accelerator="gpu", devices=[args.gpu],
    )
    train_time = time.time() - start
    log.info(f"  Training: {train_time:.1f}s")

    adata.layers["velocity"] = model.get_velocity(n_samples=25).values

    latent_time = model.get_latent_time(n_samples=25)
    adata.obs["latent_time"] = latent_time.mean(axis=1)

    _orig_inference = _velovi_mod.VELOVAE.inference
    def _patched_inference(self, *a, **kw):
        out = _orig_inference(self, *a, **kw)
        if "qz_m" in out and "qzm" not in out:
            out["qzm"] = out["qz_m"]
            out["qzv"] = out["qz_v"]
            out["z"]   = out.get("z", out["qz_m"])
        return out
    _velovi_mod.VELOVAE.inference = _patched_inference
    adata.obsm["X_velovi"] = model.get_latent_representation()
    _velovi_mod.VELOVAE.inference = _orig_inference

    scv.tl.velocity_graph(adata, vkey="velocity")

    s_fit, u_fit = model.get_expression_fit(n_samples=10)
    s_hat = s_fit.values if hasattr(s_fit, "values") else np.array(s_fit)
    u_hat = u_fit.values if hasattr(u_fit, "values") else np.array(u_fit)

    gt_time_key = cfg.get("gt_time_key")
    metrics = compute_all_metrics(
        adata,
        velocity_key="velocity",
        latent_time_key="latent_time",
        latent_key="X_velovi",
        cluster_key=cfg["cluster_key"],
        umap_key=cfg["embedding_key"],
        s_hat=s_hat, u_hat=u_hat,
        gt_time_key=gt_time_key,
    )
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
        from scipy.stats import spearmanr
        gt   = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
        pred = adata.obs["latent_time"]
        corr, _ = spearmanr(pred, gt)
        metrics["gt_pseudotime_spearman"] = float(abs(corr))
        log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["model"] = "VeloVI"
    return adata, metrics


def run_scvelo_stochastic(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: scVelo Stochastic")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    start = time.time()
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)
    train_time = time.time() - start
    log.info(f"  Time: {train_time:.1f}s")

    adata.obs["latent_time"] = adata.obs["velocity_pseudotime"]

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    adata.obsm["X_latent"] = adata.obsm["X_pca"]

    gt_time_key = cfg.get("gt_time_key")
    metrics = compute_all_metrics(
        adata,
        velocity_key="velocity",
        latent_time_key="latent_time",
        latent_key="X_latent",
        cluster_key=cfg["cluster_key"],
        umap_key=cfg["embedding_key"],
        gt_time_key=gt_time_key,
    )
    # Ground-truth pseudotime correlation if available
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
        from scipy.stats import spearmanr
        gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
        pred = adata.obs["latent_time"]
        corr, _ = spearmanr(pred, gt)
        metrics["gt_pseudotime_spearman"] = float(abs(corr))
        log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["model"] = "scVelo_stc"
    return adata, metrics


def run_unitvelo(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: UniTVelo")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        import os as _os
        _os.environ["TF_USE_LEGACY_KERAS"] = "True"
        import unitvelo as utv
    except ImportError:
        log.error("  unitvelo not installed. Skipping.")
        return None, None

    try:
        config = utv.config.Configuration()
        config.R2_ADJUST = True
        config.IROOT = None
        config.FIT_OPTION = "1"
        config.AGENES_R2 = 1

        start = time.time()
        adata = utv.run_model(adata, cfg["cluster_key"], config_file=config, normalize=False)
        train_time = time.time() - start
        log.info(f"  Training: {train_time:.1f}s")

        # Extract latent time
        latent_time = adata.obs.get(
            "unified_time", adata.obs.get("latent_time", None)
        )
        if latent_time is None:
            log.warning("  No unified_time or latent_time in adata.obs after UniTVelo.")
            latent_time = pd.Series(
                np.zeros(adata.n_obs), index=adata.obs_names
            )
        adata.obs["latent_time"] = latent_time

        # velocity is already in adata.layers["velocity"]
        scv.tl.velocity_graph(adata, vkey="velocity")

        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        adata.obsm["X_latent"] = adata.obsm["X_pca"]

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_latent",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "UniTVelo"
        return adata, metrics

    except Exception as e:
        log.error(f"  UniTVelo failed: {e}")
        return None, None


def run_celldancer(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: cellDancer")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        import celldancer as cd
        import celldancer.utilities as cdutil
        from celldancer.utilities import export_velocity_to_dynamo
    except ImportError:
        log.error("  celldancer not installed. Skipping.")
        return None, None

    try:
        import tempfile, os as _os
        start = time.time()

        # cellDancer needs Mu/Ms moments
        if "Mu" not in adata.layers or "Ms" not in adata.layers:
            log.warning("  cellDancer requires Mu/Ms layers. Skipping.")
            return None, None

        embed_key = cfg["embedding_key"]
        save_csv = os.path.join(
            cfg.get("output_dir", "outputs/celldancer_tmp"), "cell_type_u_s.csv"
        )
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)

        input_data = cdutil.adata_to_df_with_embed(
            adata,
            us_para=["Mu", "Ms"],
            cell_type_para=cfg["cluster_key"],
            embed_para=embed_key,
            save_path=save_csv,
        )
        gene_list = np.array(input_data["gene_name"].unique())
        loss_df, cellDancer_df = cd.velocity(
            input_data,
            gene_list=gene_list,
            permutation_ratio=0.125,
            n_jobs=4,
        )
        cellDancer_df = cd.compute_cell_velocity(
            cellDancer_df=cellDancer_df,
            projection_neighbor_choice="gene",
            expression_scale="power10",
            projection_neighbor_size=10,
            speed_up=(100, 100),
        )
        adata = export_velocity_to_dynamo(cellDancer_df, adata)
        adata.layers["velocity"] = adata.layers.pop("velocity_S")
        train_time = time.time() - start
        log.info(f"  Velocity: {train_time:.1f}s")

        # Pseudotime via simulation (optional, may be slow)
        try:
            cellDancer_df = cd.pseudo_time(
                cellDancer_df=cellDancer_df,
                grid=(30, 30),
                dt=0.001,
                t_total=10000,
                n_repeats=5,
                speed_up=(60, 60),
                n_paths=2,
                psrng_seeds_diffusion=list(range(5)),
                n_jobs=4,
            )
            subdf = cellDancer_df.iloc[: adata.shape[0], :]
            subdf.index = adata.obs.index
            adata.obs["latent_time"] = subdf["pseudotime"].values
        except Exception as te:
            log.warning(f"  cellDancer pseudo_time failed ({te}), using velocity_pseudotime")
            scv.tl.velocity_graph(adata, vkey="velocity")
            scv.tl.velocity_pseudotime(adata)
            adata.obs["latent_time"] = adata.obs["velocity_pseudotime"]

        scv.tl.velocity_graph(adata, vkey="velocity")

        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        adata.obsm["X_latent"] = adata.obsm["X_pca"]

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_latent",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "cellDancer"
        return adata, metrics

    except Exception as e:
        log.error(f"  cellDancer failed: {e}")
        import traceback; log.error(traceback.format_exc())
        return None, None


def run_pyrovelocity(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: PyroVelocity")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        from pyrovelocity.tasks.train import train_model
    except ImportError:
        log.error("  pyrovelocity not installed. Skipping.")
        return None, None

    try:
        import os as _os
        _os.environ.setdefault("PYROVELOCITY_TESTING_FLAG", "False")
        _os.environ.setdefault("PYROVELOCITY_LOG_LEVEL", "ERROR")
        # Pin JAX to the same GPU as PyTorch; must be set before jax initializes
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import scipy.sparse as _sp

        # pyrovelocity.setup_anndata() requires u_lib_size_raw / s_lib_size_raw in obs
        if "u_lib_size_raw" not in adata.obs:
            for obs_key, layer_keys in [
                ("u_lib_size_raw", ["unspliced", "Mu"]),
                ("s_lib_size_raw", ["spliced",   "Ms"]),
            ]:
                for lk in layer_keys:
                    if lk in adata.layers:
                        arr = adata.layers[lk]
                        if _sp.issparse(arr):
                            arr = arr.toarray()
                        adata.obs[obs_key] = np.asarray(arr).sum(axis=1).flatten()
                        break

        start = time.time()
        adata_model_pos = train_model(
            adata,
            guide_type="auto_t0_constraint",
            model_type="auto",
            use_gpu="auto",
            likelihood="Poisson",
            num_samples=30,
            log_every=100,
            learning_rate=0.01,
            seed=42,
            patient_improve=1e-3,
            max_epochs=1000,
            offset=False,
            library_size=True,
            patient_init=45,
            batch_size=4000,
            include_prior=True,
        )
        train_time = time.time() - start
        log.info(f"  Training: {train_time:.1f}s")

        pos = adata_model_pos[2]

        # Reconstruct spliced posterior mean for velocity_graph xkey
        ut = pos["ut"]
        st = pos["st"]
        adata.layers["spliced_pyro"] = np.squeeze(st.mean(0))

        if "u_scale" in pos:
            adata.layers["velocity"] = np.squeeze(
                (ut * pos["beta"] / pos["u_scale"] - st * pos["gamma"]).mean(0)
            )
        else:
            adata.layers["velocity"] = np.squeeze(
                (ut * pos["beta"] - st * pos["gamma"]).mean(0)
            )

        # Cell time
        cell_time = np.squeeze(pos["cell_time"], axis=-1)
        adata.obs["latent_time"] = np.mean(cell_time, axis=0)

        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        sc.pp.neighbors(adata, use_rep="X_pca")
        scv.tl.velocity_graph(adata, vkey="velocity", xkey="spliced_pyro")
        adata.obsm["X_latent"] = adata.obsm["X_pca"]

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_latent",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "PyroVelocity"
        return adata, metrics

    except Exception as e:
        log.error(f"  PyroVelocity failed: {e}")
        import traceback; log.error(traceback.format_exc())
        return None, None


def run_sctour(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: scTour")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        import sctour as sct
        import anndata as ad
    except ImportError:
        log.error("  sctour not installed. Skipping.")
        return None, None

    try:
        # scTour needs log1p(normalized) in .X — leave .X as-is from preprocessing
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        start = time.time()
        tnode = sct.train.Trainer(
            adata,
            loss_mode="mse",
            alpha_recon_lec=0.5,
            alpha_recon_lode=0.5,
            batch_norm=False,
        )
        tnode.train()
        train_time = time.time() - start
        log.info(f"  Training: {train_time:.1f}s")

        adata.obs["latent_time"] = tnode.get_time()
        mix_zs, zs, pred_zs = tnode.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
        adata.obsm["X_TNODE"] = mix_zs
        adata.obsm["X_VF"] = tnode.get_vector_field(
            adata.obs["latent_time"].values, mix_zs
        )

        # Velocity is in latent space — build a mini-adata for velocity_graph
        mini = ad.AnnData(X=mix_zs.copy())
        mini.layers["spliced"] = mix_zs.copy()
        mini.layers["unspliced"] = mix_zs.copy()
        mini.layers["velocity"] = adata.obsm["X_VF"].copy()
        mini.obs.index = adata.obs.index
        if "connectivities" in adata.obsp:
            mini.obsp["connectivities"] = adata.obsp["connectivities"]
            mini.obsp["distances"] = adata.obsp["distances"]
            mini.uns["neighbors"] = adata.uns.get("neighbors", {})
        else:
            sc.pp.neighbors(mini, use_rep="X", n_neighbors=30)
        scv.tl.velocity_graph(mini, vkey="velocity", xkey="spliced")
        adata.uns["velocity_graph"]     = mini.uns["velocity_graph"]
        adata.uns["velocity_graph_neg"] = mini.uns.get("velocity_graph_neg",
                                                        mini.uns["velocity_graph"] * 0)

        # scTour velocity is in latent space (n_latent_dim), not gene space.
        # Use zeros for gene-space layer; velocity_graph already transferred from mini-adata.
        adata.layers["velocity"] = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_TNODE",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "scTour"
        return adata, metrics

    except Exception as e:
        log.error(f"  scTour failed: {e}")
        import traceback; log.error(traceback.format_exc())
        return None, None


def run_cell2fate(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: cell2fate")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        import cell2fate as c2f
        import scipy.sparse as sp_sparse
    except ImportError:
        log.error("  cell2fate not installed. Skipping.")
        return None, None

    try:
        # cell2fate requires raw count layers
        for src, dst in [("spliced", "raw_spliced"), ("unspliced", "raw_unspliced")]:
            if dst not in adata.layers and src in adata.layers:
                arr = adata.layers[src]
                if sp_sparse.issparse(arr):
                    arr = np.array(arr.toarray(), dtype=np.float32)
                else:
                    arr = np.array(arr, dtype=np.float32)
                adata.layers[dst] = arr
            elif dst in adata.layers and sp_sparse.issparse(adata.layers[dst]):
                adata.layers[dst] = np.array(
                    adata.layers[dst].toarray(), dtype=np.float32
                )

        if "raw_spliced" not in adata.layers:
            log.warning("  cell2fate: no raw_spliced layer available. Skipping.")
            return None, None

        n_clusters = len(adata.obs[cfg["cluster_key"]].unique())
        n_modules = int(n_clusters * 1.15)
        log.info(f"  n_modules={n_modules} (clusters={n_clusters})")

        c2f.Cell2fate_DynamicalModel.setup_anndata(
            adata, spliced_label="raw_spliced", unspliced_label="raw_unspliced"
        )
        mod = c2f.Cell2fate_DynamicalModel(
            adata,
            n_modules=n_modules,
            Tmax_prior={"mean": 50.0, "sd": 50.0},
        )

        start = time.time()
        mod.train()
        train_time = time.time() - start
        log.info(f"  Training: {train_time:.1f}s")

        adata = mod.export_posterior(
            adata,
            sample_kwargs={
                "num_samples": 30,
                "batch_size": None,
                "use_gpu": True,
                "return_samples": False,
            },
        )
        mod.compute_and_plot_total_velocity(adata, save=False, delete=False, plot=False)

        vel = adata.layers["Velocity"]
        adata.layers["velocity"] = vel.numpy() if hasattr(vel, "numpy") else np.array(vel)
        del adata.layers["Velocity"]

        if "Time (hours)" in adata.obs:
            adata.obs["latent_time"] = adata.obs["Time (hours)"]
        else:
            log.warning("  'Time (hours)' not in obs; using velocity_pseudotime as fallback")
            scv.tl.velocity_graph(adata, vkey="velocity")
            scv.tl.velocity_pseudotime(adata)
            adata.obs["latent_time"] = adata.obs["velocity_pseudotime"]

        scv.tl.velocity_graph(adata, vkey="velocity")

        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        adata.obsm["X_latent"] = adata.obsm["X_pca"]

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_latent",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "cell2fate"
        return adata, metrics

    except Exception as e:
        log.error(f"  cell2fate failed: {e}")
        import traceback; log.error(traceback.format_exc())
        return None, None


def run_tfvelo(adata_raw, cfg, args):
    log.info("\n" + "=" * 70)
    log.info("  MODEL: TFvelo")
    log.info("=" * 70)

    adata = adata_raw.copy()
    log.info(f"  Shape: {adata.shape}")

    try:
        import sys, os as _os
        tfvelo_root = args.tf_data_dir  # e.g. .../TFvelo
        if tfvelo_root not in sys.path:
            sys.path.insert(0, tfvelo_root)
        import TFvelo as TFv

        # M_total = smoothed total (spliced + unspliced moments)
        adata.layers["M_total"] = adata.layers["Ms"] + adata.layers["Mu"]

        # get_TFs uses hardcoded relative paths (ENCODE/processed/) —
        # must run from the TFvelo root directory.
        # ENCODE uses human uppercase symbols; temporarily uppercase gene names
        # so mouse datasets (e.g. Actb -> ACTB) match the database.
        orig_var_names = adata.var_names.tolist()
        adata.var_names = adata.var_names.str.upper()
        adata.layers["M_total"] = adata.layers["Ms"] + adata.layers["Mu"]

        orig_cwd = _os.getcwd()
        _os.chdir(tfvelo_root)
        try:
            log.info("  Building TF-target regulatory graph (ENCODE)...")
            TFv.pp.get_TFs(adata, databases=["ENCODE"])
        finally:
            _os.chdir(orig_cwd)

        # Restore original gene names before any downstream scvelo calls
        adata.var_names = orig_var_names

        log.info(f"  TFs assigned. n_TFs per gene: mean={adata.var['n_TFs'].mean():.1f}")

        start = time.time()
        log.info("  Running TFvelo recover_dynamics (EM, n_jobs=1)...")
        TFv.tl.recover_dynamics(adata, var_names="all", n_jobs=1)
        train_time = time.time() - start
        log.info(f"  recover_dynamics done: {train_time:.1f}s")

        # velo_hat is the TF-regulated velocity stored by write_result
        if "velo_hat" not in adata.layers:
            log.error("  velo_hat layer missing after recover_dynamics")
            return None, None
        vel = adata.layers["velo_hat"]
        if hasattr(vel, "toarray"):
            vel = vel.toarray()
        vel = np.nan_to_num(np.array(vel, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if vel.size == 0 or (vel == 0).all():
            log.error("  velo_hat is empty/all-zero after recover_dynamics (likely no TF matches)")
            return None, None
        adata.layers["velocity"] = vel

        # Downstream: velocity graph + latent time (use scvelo on gene-space velocity)
        scv.tl.velocity_graph(adata, vkey="velocity")
        scv.tl.latent_time(adata)
        adata.obs["latent_time"] = adata.obs["latent_time"]

        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        adata.obsm["X_latent"] = adata.obsm["X_pca"]

        gt_time_key = cfg.get("gt_time_key")
        metrics = compute_all_metrics(
            adata,
            velocity_key="velocity",
            latent_time_key="latent_time",
            latent_key="X_latent",
            cluster_key=cfg["cluster_key"],
            umap_key=cfg["embedding_key"],
            gt_time_key=gt_time_key,
        )
        if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
            from scipy.stats import spearmanr
            gt = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
            pred = adata.obs["latent_time"]
            corr, _ = spearmanr(pred, gt)
            metrics["gt_pseudotime_spearman"] = float(abs(corr))
            log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

        metrics["train_time_seconds"] = round(train_time, 1)
        metrics["model"] = "TFvelo"
        return adata, metrics

    except Exception as e:
        log.error(f"  TFvelo failed: {e}")
        import traceback; log.error(traceback.format_exc())
        return None, None


def run_tarvi(adata_raw, mode, cfg, args):
    mcfg = TARVI_MODE_CONFIG[mode]
    use_tf   = mcfg["use_tf"]
    ode_w    = mcfg["ode_residual_weight"]
    vpt_blend = mcfg["vpt_blend"]
    use_vpt  = mcfg["use_vpt"]

    log.info("\n" + "=" * 70)
    log.info(f"  MODEL: TARVI [{mode}]  TF={use_tf}  ODE_res={ode_w}  VPT={vpt_blend}")
    log.info("=" * 70)

    adata = adata_raw.copy()
    adata = preprocess_data(adata)
    log.info(f"  Shape after preprocess: {adata.shape}")

    TARVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    model = TARVI(
        adata,
        n_hidden=256,
        n_latent=10,
        use_tf_regulation=use_tf,
        tf_data_dir=args.tf_data_dir,
        tf_l1_weight=0.01,
        velocity_consistency_weight=0.2,
        temporal_smoothness_weight=0.1,
        use_learned_time=True,
        time_reg_weight=1.0,
        use_gnn_refinement=False,
        ode_residual_weight=ode_w,
        velo_confidence_weight=0.0,
    )
    log.info(f"  {model._model_summary_string}")

    start = time.time()
    model.train(
        max_epochs=args.max_epochs, lr=1e-2,
        batch_size=min(256, max(32, adata.n_obs // 4)),
        accelerator="gpu", devices=[args.gpu],
    )
    train_time = time.time() - start
    log.info(f"  Training: {train_time:.1f}s")

    adata.layers["velocity"] = model.get_velocity(n_samples=25, smooth=False, refine=False).values
    adata.obsm["X_tarvi"]     = model.get_latent_representation()
    scv.tl.velocity_graph(adata, vkey="velocity")

    if use_vpt:
        log.info(f"  Applying VPT blend={vpt_blend}...")
        adata.obs["latent_time"] = model.get_velocity_pseudotime(
            adata=adata, vkey="velocity", blend=vpt_blend, n_samples=25
        )
    else:
        latent_time = model.get_latent_time(n_samples=25)
        ode_time = (latent_time.mean(axis=1).values
                    if hasattr(latent_time, "values") else latent_time.mean(axis=1))
        ode_min, ode_max = ode_time.min(), ode_time.max()
        ode_time_norm = (ode_time - ode_min) / (ode_max - ode_min + 1e-8)
        if model.module.use_learned_time:
            learned_t = model.get_learned_time(n_samples=25)
            from scipy.stats import spearmanr
            corr, _ = spearmanr(learned_t, ode_time_norm)
            if corr < 0:
                learned_t = 1.0 - learned_t
            adata.obs["latent_time"] = 0.7 * learned_t + 0.3 * ode_time_norm
        else:
            adata.obs["latent_time"] = ode_time_norm

    s_fit, u_fit = model.get_expression_fit(n_samples=10)
    s_hat = s_fit.values if hasattr(s_fit, "values") else np.array(s_fit)
    u_hat = u_fit.values if hasattr(u_fit, "values") else np.array(u_fit)

    gt_time_key = cfg.get("gt_time_key")
    metrics = compute_all_metrics(
        adata,
        velocity_key="velocity",
        latent_time_key="latent_time",
        latent_key="X_tarvi",
        cluster_key=cfg["cluster_key"],
        umap_key=cfg["embedding_key"],
        s_hat=s_hat, u_hat=u_hat,
        gt_time_key=gt_time_key,
    )
    if cfg["gt_pseudotime_key"] and cfg["gt_pseudotime_key"] in adata_raw.obs:
        from scipy.stats import spearmanr
        gt   = adata_raw.obs[cfg["gt_pseudotime_key"]].loc[adata.obs_names]
        pred = adata.obs["latent_time"]
        corr, _ = spearmanr(pred, gt)
        metrics["gt_pseudotime_spearman"] = float(abs(corr))
        log.info(f"  GT-pseudotime Spearman: {corr:.4f} (|corr|={abs(corr):.4f})")

    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["model"] = f"TARVI_{mode}"
    return adata, metrics


# ─── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(all_adata, all_metrics, cfg, dataset_name, output_dir):
    models = list(all_adata.items())
    n = len(models)
    basis = cfg["basis"]
    emb_key = cfg["embedding_key"]
    cluster_key = cfg["cluster_key"]
    title_prefix = cfg["title"]
    dot_size_stream  = cfg.get("dot_size_stream", 60)
    dot_size_scatter = cfg.get("dot_size_scatter", 20)

    # Consistent cluster colors
    ref_adata = models[0][1]
    if hasattr(ref_adata.obs[cluster_key], "cat"):
        clusters = ref_adata.obs[cluster_key].cat.categories.tolist()
    else:
        clusters = sorted(ref_adata.obs[cluster_key].unique().tolist())
    cluster_colors = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                      for i, c in enumerate(clusters)}

    for _, adata in models:
        adata.uns[f"{cluster_key}_colors"] = [cluster_colors[c] for c in clusters]
        if "velocity_graph" not in adata.uns:
            try:
                scv.tl.velocity_graph(adata, vkey="velocity")
            except Exception:
                pass

    # ── 3-row combined figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 13), facecolor="white")
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, (name, adata) in enumerate(models):
        label = name.replace("_", " ")

        ax = axes[0, col]
        try:
            scv.pl.velocity_embedding_stream(
                adata, basis=basis, vkey="velocity",
                color=cluster_key, ax=ax, show=False,
                legend_loc="none", title="",
                size=dot_size_stream, alpha=0.7, linewidth=0.8, arrow_size=1.2,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Stream failed:\n{str(e)[:50]}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Velocity\nStreamlines", fontsize=11, fontweight="bold")
        ax.set_xlabel("")

        ax = axes[1, col]
        if "latent_time" in adata.obs:
            lt = adata.obs["latent_time"].values
            xy = adata.obsm[emb_key]
            scp = ax.scatter(xy[:, 0], xy[:, 1], c=lt, cmap="viridis",
                             s=dot_size_scatter, alpha=0.85, rasterized=True)
            plt.colorbar(scp, ax=ax, shrink=0.6, label="Latent Time")
        else:
            ax.text(0.5, 0.5, "No latent_time", ha="center", va="center",
                    transform=ax.transAxes)
        if col == 0:
            ax.set_ylabel("Latent\nTime", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2, col]
        xy = adata.obsm[emb_key]
        for c in clusters:
            mask = adata.obs[cluster_key].values == c
            if mask.sum() > 0:
                ax.scatter(xy[mask, 0], xy[mask, 1], c=cluster_colors[c],
                           s=dot_size_scatter, alpha=0.85, label=c, rasterized=True)
        if col == 0:
            ax.set_ylabel("Cell Type\nClusters", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.scatter([], [], c=cluster_colors[c], s=40, label=c)
               for c in clusters]
    fig.legend(handles=handles, labels=clusters, loc="lower center",
               ncol=min(6, len(clusters)), fontsize=9, markerscale=1.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"TARVI — {title_prefix} ({basis.upper()})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_combined.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"  Saved: {path}")

    # ── Individual stream plots ───────────────────────────────────────────────
    for name, adata in models:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        try:
            scv.pl.velocity_embedding_stream(
                adata, basis=basis, vkey="velocity", color=cluster_key,
                ax=ax, show=False, title=f"{name} — Velocity ({basis.upper()})",
                size=dot_size_stream, alpha=0.8, linewidth=1.0, legend_loc="right margin",
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed: {e}", ha="center", va="center",
                    transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"velocity_stream_{name}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()

    # ── Individual latent time plots ──────────────────────────────────────────
    for name, adata in models:
        if "latent_time" not in adata.obs:
            continue
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        lt = adata.obs["latent_time"].values
        xy = adata.obsm[emb_key]
        scp = ax.scatter(xy[:, 0], xy[:, 1], c=lt, cmap="viridis",
                         s=dot_size_scatter, alpha=0.9, rasterized=True)
        plt.colorbar(scp, ax=ax, shrink=0.7, label="Latent Time")
        ax.set_title(f"{name} — Latent Time ({basis.upper()})",
                     fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"latent_time_{name}.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()

    # ── Radar plot ────────────────────────────────────────────────────────────
    radar_metrics = [
        ("iccoh_overall",               "ICCoH"),
        ("cbdir_overall",               "CBDir"),
        ("velocity_confidence_mean",    "Vel Conf"),
        ("pseudotime_spearman_2d",      "PT Spear"),
        ("pseudotime_local_consistency","PT Consist"),
        ("root_cell_accuracy",          "Root Acc"),
        ("transition_probability_score","Trans Prob"),
        ("self_transition_score",       "STS"),
        ("normalized_transition_entropy","NTE"),
    ]
    # Add GT pseudotime axis if available for this dataset
    if cfg["gt_pseudotime_key"]:
        radar_metrics.append(("gt_pseudotime_spearman", "GT PT Spear"))

    cat_keys   = [m[0] for m in radar_metrics]
    categories = [m[1] for m in radar_metrics]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    model_colors = {
        "velocyto":      "#FF5722",
        "scVelo_dyn":    "#9E9E9E",
        "scVelo_stc":    "#607D8B",
        "VeloVI":        "#2196F3",
        "UniTVelo":      "#FF9800",
        "cellDancer":    "#9C27B0",
        "scTour":        "#009688",
        "TFvelo":        "#3F51B5",
        "cell2fate":     "#795548",
        "TARVI_baseline": "#90CAF9",
        "TARVI_tf":       "#4CAF50",
        "TARVI_full":     "#E91E63",
    }
    model_labels = {
        "velocyto":      "velocyto",
        "scVelo_dyn":    "scVelo Dyn",
        "scVelo_stc":    "scVelo Stc",
        "VeloVI":        "VeloVI",
        "UniTVelo":      "UniTVelo",
        "cellDancer":    "cellDancer",
        "scTour":        "scTour",
        "TFvelo":        "TFvelo",
        "cell2fate":     "cell2fate",
        "TARVI_baseline": "TARVI (base)",
        "TARVI_tf":       "TARVI (TF)",
        "TARVI_full":     "TARVI (full)",
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="white")
    for mkey, metrics in all_metrics.items():
        vals = []
        for k in cat_keys:
            v = metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(0.0)
            else:
                vals.append(float(np.clip(v, 0, 1)))
        vals += vals[:1]
        color = model_colors.get(mkey, "#999999")
        label = model_labels.get(mkey, mkey)
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"{title_prefix} — Model Comparison Radar",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10,
              framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"  Saved: {path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    models_list = list(all_metrics.keys())
    n_m = len(models_list)
    x = np.arange(len(radar_metrics))
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    for i, mkey in enumerate(models_list):
        vals = []
        for k, _ in radar_metrics:
            v = all_metrics[mkey].get(k)
            vals.append(
                float(np.clip(v, 0, 1))
                if v is not None and not (isinstance(v, float) and np.isnan(v))
                else 0.0
            )
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model_labels.get(mkey, mkey),
               color=model_colors.get(mkey, "#999"),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in radar_metrics], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{title_prefix} — Model Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"  Saved: {path}")

    log.info(f"  All plots saved to {output_dir}")


# ─── Per-model save/load (resume support) ─────────────────────────────────────

def _save_model(name, adata, metrics, output_dir):
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    adata.write_h5ad(os.path.join(model_dir, "results.h5ad"))
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in metrics.items()}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(clean, f, indent=2)
    log.info(f"  [saved] {model_dir}/")


def _load_model_if_done(name, output_dir):
    """Return (adata, metrics) from disk if already completed, else (None, None)."""
    model_dir = os.path.join(output_dir, name)
    h5ad = os.path.join(model_dir, "results.h5ad")
    mjson = os.path.join(model_dir, "metrics.json")
    if os.path.isfile(h5ad) and os.path.isfile(mjson):
        try:
            adata = sc.read_h5ad(h5ad)
            with open(mjson) as f:
                metrics = json.load(f)
            log.info(f"  [resume] Loaded {name} from {model_dir}/")
            return adata, metrics
        except Exception as e:
            log.warning(f"  [resume] Failed to load {name}: {e}")
    return None, None


# ─── Comparison table ─────────────────────────────────────────────────────────

def print_comparison(all_metrics, cfg, dataset_name, output_dir):
    models = list(all_metrics.keys())
    key_metrics = [
        ("iccoh_overall",               "ICCoH",               True),
        ("cbdir_overall",               "CBDir",               True),
        ("velocity_confidence_mean",    "Vel Confidence",      True),
        ("velocity_magnitude_corr",     "Vel-Expr Corr",       True),
        ("pseudotime_spearman_2d",      "PT Spearman (2D)",    True),
        ("pseudotime_local_consistency","PT Consistency",      True),
        ("root_cell_accuracy",          "Root Cell Acc",       True),
        ("gene_velocity_r2_mean",       "Gene R2 (mean)",      True),
        ("directional_cosine_sim_mean", "Dir Cosine Sim",      True),
        ("transition_probability_score","Trans Prob",          True),
        ("self_transition_score",       "STS",                 True),
        ("normalized_transition_entropy","NTE",                True),
        ("train_time_seconds",          "Train Time (s)",      None),
    ]
    if cfg["gt_pseudotime_key"]:
        key_metrics.insert(8, ("gt_pseudotime_spearman", "GT PT Spearman", True))

    # Add temporal Spearman corr when gt_time_key is present
    if cfg.get("gt_time_key") and cfg["gt_time_key"] in (
        list(next(iter(all_metrics.values())).keys()) if all_metrics else []
    ):
        key_metrics.insert(-1, ("temporal_spearman_corr", "Temporal Spear (TSC)", True))
        key_metrics.insert(-1, ("cross_boundary_correctness", "CTO", True))

    log.info(f"\n{'='*110}")
    log.info(f"{cfg['title'].upper()} — Model Comparison")
    log.info(f"{'='*110}")
    header = f"  {'Metric':<26}"
    for m in models:
        header += f" {m:>16}"
    log.info(header)
    log.info("  " + "-" * (26 + 17 * len(models)))

    best_counts = {m: 0 for m in models}
    rows = []
    for metric_key, display_name, higher_is_better in key_metrics:
        line = f"  {display_name:<26}"
        vals = {}
        row = {"Metric": display_name}
        for m in models:
            v = all_metrics[m].get(metric_key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                line += f" {'N/A':>16}"; row[m] = None
            elif isinstance(v, (int, float)):
                fv = float(v)
                fmt = f"{fv:.4f}" if abs(fv) >= 0.001 or fv == 0 else f"{fv:.2e}"
                line += f" {fmt:>16}"; vals[m] = fv; row[m] = fv
            else:
                line += f" {str(v):>16}"; row[m] = v
        if vals and higher_is_better is not None:
            best = max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)
            best_counts[best] = best_counts.get(best, 0) + 1
        log.info(line)
        rows.append(row)

    log.info("\n  Wins per model:")
    for m, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {m}: {count} best metrics")

    log.info("\n  Note: STS (Self-Transition Score) and NTE (Normalized Transition Entropy)")
    log.info("        are higher-is-better for robust/uncertain models (e.g. PyroVelocity, scTour).")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{dataset_name}_comparison.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"\nCSV: {csv_path}")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TARVI on multiple RNA velocity benchmark datasets"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["dentate_gyrus", "bonemarrow", "forebrain",
                                 "gastrulation", "reprogramming", "pbmc68k",
                                 "zebrafish", "sceu_organoid", "chromaffin",
                                 "pancreas"],
                        help="Which dataset to run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--tf_data_dir", type=str,
                        default="./data/TFvelo")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override default output dir")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tarvi_modes", type=str, nargs="+",
                        default=["baseline", "tf", "full"])
    parser.add_argument("--skip_velocyto",       action="store_true", default=False)
    parser.add_argument("--skip_scvelo",        action="store_true", default=False)
    parser.add_argument("--skip_scvelo_stc",    action="store_true", default=False)
    parser.add_argument("--skip_velovi",        action="store_true", default=False)
    parser.add_argument("--skip_unitvelo",      action="store_true", default=False)
    parser.add_argument("--skip_celldancer",    action="store_true", default=False)
    parser.add_argument("--skip_pyrovelocity",  action="store_true", default=True)
    parser.add_argument("--skip_sctour",        action="store_true", default=False)
    parser.add_argument("--skip_cell2fate",     action="store_true", default=False)
    parser.add_argument("--skip_tfvelo",        action="store_true", default=False)
    # Precomputed results from standalone runners (separate conda envs)
    parser.add_argument("--celldancer_results", type=str, default=None,
                        help="Dir with results.h5ad + meta.json from run_celldancer_standalone.py")
    parser.add_argument("--cell2fate_results",  type=str, default=None,
                        help="Dir with results.h5ad + meta.json from run_cell2fate_standalone.py")
    # Save preprocessed h5ad for use by standalone runners
    parser.add_argument("--save_preprocessed",  action="store_true", default=False,
                        help="Save preprocessed adata to output_dir/preprocessed.h5ad then exit")
    parser.add_argument("--resume",             action="store_true", default=False,
                        help="Load already-completed model results from disk and skip re-running them")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    output_dir = args.output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(args.log)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        log.info(f"GPU: cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

    log.info("=" * 70)
    log.info(f"TARVI — {cfg['title']}")
    log.info("=" * 70)
    log.info(f"  n_top_genes={args.n_top_genes}  n_neighbors={args.n_neighbors}")
    log.info(f"  max_epochs={args.max_epochs}")
    log.info(f"  basis={cfg['basis']}  embedding={cfg['embedding_key']}")

    adata_raw, cfg = load_dataset(args.dataset, args)
    log.info(f"\nFinal dataset: {adata_raw.n_obs} cells x {adata_raw.n_vars} genes")

    # Save preprocessed h5ad for standalone runners (separate envs) then exit
    if args.save_preprocessed:
        prep_path = os.path.join(output_dir, "preprocessed.h5ad")
        adata_raw.write_h5ad(prep_path)
        log.info(f"Saved preprocessed: {prep_path}")
        log.info("Run standalone runners, then re-run without --save_preprocessed.")
        return

    all_metrics = {}
    all_adata   = {}

    def _run_or_load(name, skip_flag, run_fn, *run_args, seed_torch=False):
        if skip_flag:
            return
        if args.resume:
            adata_r, metrics_r = _load_model_if_done(name, output_dir)
            if adata_r is not None:
                all_metrics[name] = metrics_r
                all_adata[name]   = adata_r
                return
        np.random.seed(args.seed)
        if seed_torch:
            torch.manual_seed(args.seed)
        adata_r, metrics_r = run_fn(*run_args)
        if metrics_r:
            all_metrics[name] = metrics_r
            all_adata[name]   = adata_r
            _save_model(name, adata_r, metrics_r, output_dir)

    # 0. velocyto
    _run_or_load("velocyto",   args.skip_velocyto,  run_velocyto,          adata_raw, cfg, args)
    # 1. scVelo dynamical
    _run_or_load("scVelo_dyn", args.skip_scvelo,     run_scvelo_dynamical,  adata_raw, cfg, args)
    # 2. scVelo stochastic
    _run_or_load("scVelo_stc", args.skip_scvelo_stc, run_scvelo_stochastic, adata_raw, cfg, args)
    # 3. VeloVI
    _run_or_load("VeloVI",     args.skip_velovi,     run_velovi,            adata_raw, cfg, args, seed_torch=True)
    # 4. UniTVelo
    _run_or_load("UniTVelo",   args.skip_unitvelo,   run_unitvelo,          adata_raw, cfg, args, seed_torch=True)

    # 5. cellDancer (in-process or load precomputed from separate celldancer_env)
    if args.celldancer_results:
        h5ad_path = os.path.join(args.celldancer_results, "results.h5ad")
        meta_path = os.path.join(args.celldancer_results, "meta.json")
        if os.path.isfile(h5ad_path):
            log.info("\n" + "=" * 70)
            log.info("  MODEL: cellDancer [loading precomputed]")
            log.info("=" * 70)
            adata_cd = sc.read_h5ad(h5ad_path)
            meta_cd  = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
            scv.tl.velocity_graph(adata_cd, vkey="velocity")
            if "X_pca" not in adata_cd.obsm:
                sc.tl.pca(adata_cd)
            adata_cd.obsm["X_latent"] = adata_cd.obsm["X_pca"]
            gt_time_key = cfg.get("gt_time_key")
            metrics_cd = compute_all_metrics(
                adata_cd, velocity_key="velocity", latent_time_key="latent_time",
                latent_key="X_latent", cluster_key=cfg["cluster_key"],
                umap_key=cfg["embedding_key"], gt_time_key=gt_time_key,
            )
            metrics_cd["train_time_seconds"] = meta_cd.get("train_time_seconds", 0.0)
            metrics_cd["model"] = "cellDancer"
            all_metrics["cellDancer"] = metrics_cd
            all_adata["cellDancer"]   = adata_cd
            log.info(f"  Loaded precomputed cellDancer from {args.celldancer_results}")
        else:
            log.warning(f"  --celldancer_results: no results.h5ad in {args.celldancer_results}")
    elif not args.skip_celldancer:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        adata_cd, metrics_cd = run_celldancer(adata_raw, cfg, args)
        if metrics_cd:
            all_metrics["cellDancer"] = metrics_cd
            all_adata["cellDancer"]   = adata_cd

    # 6. PyroVelocity
    _run_or_load("PyroVelocity", args.skip_pyrovelocity, run_pyrovelocity, adata_raw, cfg, args, seed_torch=True)

    # 7. scTour
    _run_or_load("scTour", args.skip_sctour, run_sctour, adata_raw, cfg, args, seed_torch=True)

    # 8. cell2fate (in-process or load precomputed from separate cell2fate_env)
    if args.cell2fate_results:
        h5ad_path = os.path.join(args.cell2fate_results, "results.h5ad")
        meta_path = os.path.join(args.cell2fate_results, "meta.json")
        if os.path.isfile(h5ad_path):
            log.info("\n" + "=" * 70)
            log.info("  MODEL: cell2fate [loading precomputed]")
            log.info("=" * 70)
            adata_c2f = sc.read_h5ad(h5ad_path)
            meta_c2f  = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
            scv.tl.velocity_graph(adata_c2f, vkey="velocity")
            if "X_pca" not in adata_c2f.obsm:
                sc.tl.pca(adata_c2f)
            adata_c2f.obsm["X_latent"] = adata_c2f.obsm["X_pca"]
            gt_time_key = cfg.get("gt_time_key")
            metrics_c2f = compute_all_metrics(
                adata_c2f, velocity_key="velocity", latent_time_key="latent_time",
                latent_key="X_latent", cluster_key=cfg["cluster_key"],
                umap_key=cfg["embedding_key"], gt_time_key=gt_time_key,
            )
            metrics_c2f["train_time_seconds"] = meta_c2f.get("train_time_seconds", 0.0)
            metrics_c2f["model"] = "cell2fate"
            all_metrics["cell2fate"] = metrics_c2f
            all_adata["cell2fate"]   = adata_c2f
            _save_model("cell2fate", adata_c2f, metrics_c2f, output_dir)
            log.info(f"  Loaded precomputed cell2fate from {args.cell2fate_results}")
        else:
            log.warning(f"  --cell2fate_results: no results.h5ad in {args.cell2fate_results}")
    elif not args.skip_cell2fate:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        adata_c2f, metrics_c2f = run_cell2fate(adata_raw, cfg, args)
        if metrics_c2f:
            all_metrics["cell2fate"] = metrics_c2f
            all_adata["cell2fate"]   = adata_c2f

    # 9. TFvelo
    _run_or_load("TFvelo", args.skip_tfvelo, run_tfvelo, adata_raw, cfg, args)

    # 10. TARVI modes
    for mode in args.tarvi_modes:
        _run_or_load(f"TARVI_{mode}", False, run_tarvi, adata_raw, mode, cfg, args, seed_torch=True)

    # Save all metrics
    with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
        json.dump(
            {m: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in met.items()}
             for m, met in all_metrics.items()},
            f, indent=2,
        )

    # Comparison table + plots
    print_comparison(all_metrics, cfg, args.dataset, output_dir)
    generate_plots(all_adata, all_metrics, cfg, args.dataset, output_dir)

    log.info(f"\nDone! All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
