"""Training script for TARVI (Transcription-Factor Aided RNA Velocity Inference).

Supports any scRNA-seq dataset with spliced/unspliced layers.

Usage examples:
    # Pancreas (built-in scvelo dataset)
    python train.py --dataset pancreas --mode baseline --log outputs/baseline.log

    # Custom dataset from h5ad file
    python train.py --data_path /path/to/data.h5ad --mode tf --log outputs/tf_run.log

    # Full mode on GPU 0
    python train.py --dataset pancreas --mode full --gpu 0 --log outputs/full.log
"""

import argparse
import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tmux/headless
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvelo as scv
import torch

# Add tarvi to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tarvi import TARVI, preprocess_data


# Module-level logger
log = logging.getLogger("tarvi.train")


def setup_logging(log_file: str):
    """Setup dual logging to file and console."""
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers
    root.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    log.info(f"Logging to: {os.path.abspath(log_file)}")


# ---- Built-in dataset loaders ----
BUILTIN_DATASETS = {
    "pancreas": scv.datasets.pancreas,
    "dentategyrus": scv.datasets.dentategyrus,
    "forebrain": scv.datasets.forebrain,
    "bonemarrow": scv.datasets.bonemarrow,
}


def load_dataset(args):
    """Load dataset from either a built-in name or a file path."""
    if args.data_path is not None:
        log.info(f"Loading dataset from: {args.data_path}")
        adata = sc.read_h5ad(args.data_path)
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    elif args.dataset in BUILTIN_DATASETS:
        log.info(f"Loading built-in dataset: {args.dataset}")
        adata = BUILTIN_DATASETS[args.dataset]()
        dataset_name = args.dataset
    else:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. "
            f"Use --data_path for custom datasets or choose from: {list(BUILTIN_DATASETS.keys())}"
        )

    log.info(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    log.info(f"  Layers: {list(adata.layers.keys())}")
    return adata, dataset_name


def preprocess(adata, args):
    """Standard scRNA-seq velocity preprocessing pipeline."""
    log.info("Preprocessing...")

    has_Ms = "Ms" in adata.layers

    if not ("spliced" in adata.layers or has_Ms):
        raise ValueError(
            "Dataset must have 'spliced' (or 'Ms') layer. "
            "Available layers: " + str(list(adata.layers.keys()))
        )

    if not has_Ms:
        log.info("  Filtering genes...")
        scv.pp.filter_genes(adata, min_shared_counts=args.min_shared_counts)
        log.info("  Normalizing...")
        scv.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        log.info(f"  Selecting top {args.n_top_genes} HVGs...")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, subset=True)
        log.info("  Computing moments...")
        scv.pp.moments(adata, n_pcs=30, n_neighbors=args.n_neighbors)
    else:
        log.info("  Moments already computed (Ms/Mu layers found)")
        if adata.n_vars > args.n_top_genes:
            has_hvg = "highly_variable" in adata.var and adata.var["highly_variable"].any()
            if not has_hvg:
                log.info(f"  Selecting top {args.n_top_genes} HVGs...")
                sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, subset=True)

    adata = preprocess_data(adata)
    log.info(f"  Final shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    return adata


def train_model(adata, args, dataset_name):
    """Configure and train TARVI model."""
    use_tf = args.mode in ["tf", "tf_nb", "full"]
    use_nb = args.mode in ["tf_nb", "full"]
    smooth = args.mode == "full"

    log.info(f"\n{'='*60}")
    log.info(f"TARVI Training Configuration")
    log.info(f"{'='*60}")
    log.info(f"  Dataset:            {dataset_name} ({adata.shape[0]} cells, {adata.shape[1]} genes)")
    log.info(f"  Mode:               {args.mode}")
    log.info(f"  TF regulation:      {use_tf}")
    log.info(f"  NB likelihood:      {use_nb}")
    log.info(f"  Velocity smoothing: {smooth}")
    log.info(f"  GPU:                cuda:{args.gpu}")
    log.info(f"  Epochs:             {args.max_epochs}")
    log.info(f"  Batch size:         {args.batch_size}")
    log.info(f"  Learning rate:      {args.lr}")
    log.info(f"  Latent dim:         {args.n_latent}")
    log.info(f"  Hidden dim:         {args.n_hidden}")
    log.info(f"{'='*60}\n")

    TARVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

    model = TARVI(
        adata,
        n_hidden=args.n_hidden,
        n_latent=args.n_latent,
        use_tf_regulation=use_tf,
        tf_data_dir=args.tf_data_dir,
        use_nb_likelihood=use_nb,
        tf_l1_weight=args.tf_l1_weight,
    )

    log.info(f"{model._model_summary_string}\n")

    start_time = time.time()
    model.train(
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accelerator="gpu",
        devices=[args.gpu],
    )
    train_time = time.time() - start_time
    log.info(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    return model, smooth, train_time


def extract_results(model, adata, smooth, args, dataset_name):
    """Extract velocity, latent time, and other results."""
    log.info("\nExtracting results...")

    log.info("  Velocity estimates (25 posterior samples)...")
    velocities = model.get_velocity(n_samples=25, smooth=smooth)
    adata.layers["velocity"] = velocities.values

    log.info("  Latent time...")
    latent_time = model.get_latent_time(n_samples=25)
    adata.obs["latent_time"] = latent_time.mean(axis=1)

    log.info("  Latent representation...")
    adata.obsm["X_tarvi"] = model.get_latent_representation()

    log.info("  Velocity graph...")
    scv.tl.velocity_graph(adata, vkey="velocity")

    rates = model.get_rates()
    log.info(f"\n  Kinetic rates:")
    log.info(f"    alpha: mean={rates['alpha'].mean():.3f}, std={rates['alpha'].std():.3f}")
    log.info(f"    beta:  mean={rates['beta'].mean():.3f}, std={rates['beta'].std():.3f}")
    log.info(f"    gamma: mean={rates['gamma'].mean():.3f}, std={rates['gamma'].std():.3f}")

    return rates


def save_results(model, adata, rates, args, dataset_name, train_time):
    """Save all results: h5ad, plots, TF weights, metrics."""
    output_dir = os.path.join(args.output_dir, dataset_name, args.mode)
    os.makedirs(output_dir, exist_ok=True)

    # Save adata
    h5ad_path = os.path.join(output_dir, "results.h5ad")
    adata.write_h5ad(h5ad_path)
    log.info(f"\n  Results saved to: {h5ad_path}")

    # Plots
    log.info("  Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocity", basis="umap", ax=axes[0],
        show=False, title=f"TARVI [{args.mode}] Velocity"
    )
    sc.pl.umap(adata, color="latent_time", ax=axes[1], show=False, title="Latent Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "velocity_stream.png"), dpi=150)
    plt.close()

    for col in ["clusters", "celltype", "cell_type", "louvain", "leiden"]:
        if col in adata.obs:
            fig, ax = plt.subplots(figsize=(7, 5))
            sc.pl.umap(adata, color=col, ax=ax, show=False, title=col)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"clusters_{col}.png"), dpi=150)
            plt.close()
            break

    # TF weights
    use_tf = args.mode in ["tf", "tf_nb", "full"]
    if use_tf:
        tf_weights = model.get_tf_weights()
        if tf_weights is not None:
            tf_weights.to_csv(os.path.join(output_dir, "tf_weights.csv"))
            mean_abs_weights = tf_weights.abs().mean(axis=0).sort_values(ascending=False)
            log.info(f"\n  Top 10 TFs by mean absolute weight:")
            for tf_name, weight in mean_abs_weights.head(10).items():
                log.info(f"    {tf_name}: {weight:.4f}")
            top_tfs = mean_abs_weights.head(50)
            top_tfs.to_csv(os.path.join(output_dir, "top_tfs.csv"))

    # Metrics JSON
    metrics = {
        "dataset": dataset_name,
        "mode": args.mode,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "alpha_mean": float(rates["alpha"].mean()),
        "beta_mean": float(rates["beta"].mean()),
        "gamma_mean": float(rates["gamma"].mean()),
        "max_epochs": args.max_epochs,
        "n_latent": args.n_latent,
        "n_hidden": args.n_hidden,
        "train_time_seconds": round(train_time, 1),
        "gpu": args.gpu,
        "log_file": os.path.abspath(args.log),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Copy log file to output dir
    import shutil
    shutil.copy2(os.path.abspath(args.log), os.path.join(output_dir, "training.log"))

    log.info(f"\n  All outputs saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="TARVI: Transcription-Factor Aided RNA Velocity Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --dataset pancreas --mode baseline --gpu 0 --log outputs/baseline.log
  python train.py --dataset pancreas --mode full --gpu 0 --log outputs/full.log
  python train.py --data_path /path/to/data.h5ad --mode tf --gpu 0 --log my_run.log
        """,
    )

    # Data
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--dataset",
        choices=list(BUILTIN_DATASETS.keys()),
        help="Built-in scvelo dataset name",
    )
    data_group.add_argument(
        "--data_path",
        type=str,
        help="Path to h5ad file with spliced/unspliced layers",
    )

    # Model
    parser.add_argument(
        "--mode",
        choices=["baseline", "tf", "tf_nb", "full"],
        default="baseline",
        help="Training mode: baseline | tf | tf_nb | full (default: baseline)",
    )
    parser.add_argument("--n_latent", type=int, default=10, help="Latent dimension (default: 10)")
    parser.add_argument("--n_hidden", type=int, default=256, help="Hidden layer size (default: 256)")

    # Training
    parser.add_argument("--max_epochs", type=int, default=500, help="Max training epochs (default: 500)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")

    # Preprocessing
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of HVGs (default: 2000)")
    parser.add_argument("--n_neighbors", type=int, default=30, help="Neighbors for moments (default: 30)")
    parser.add_argument("--min_shared_counts", type=int, default=30, help="Min shared counts (default: 30)")

    # TF regulation
    parser.add_argument("--tf_data_dir", type=str, default=None, help="TF database directory")
    parser.add_argument("--tf_l1_weight", type=float, default=0.01, help="TF L1 regularization (default: 0.01)")

    # Output & Logging
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory (default: outputs)")
    parser.add_argument("--log", type=str, required=True, help="Path to training log file (required)")

    args = parser.parse_args()

    # ---- Setup ----
    setup_logging(args.log)

    log.info("=" * 60)
    log.info("TARVI - Transcription-Factor Aided RNA Velocity Inference")
    log.info("=" * 60)
    log.info(f"Command: {' '.join(sys.argv)}")
    log.info(f"Working dir: {os.getcwd()}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        log.info(f"GPU: cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        log.warning("CUDA not available! Falling back to CPU.")

    # ---- Pipeline ----
    adata, dataset_name = load_dataset(args)
    adata = preprocess(adata, args)
    model, smooth, train_time = train_model(adata, args, dataset_name)
    rates = extract_results(model, adata, smooth, args, dataset_name)
    save_results(model, adata, rates, args, dataset_name, train_time)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
