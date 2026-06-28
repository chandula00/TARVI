"""Transcription Factor Regulation Module for TARVI."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData


class TFRegulatedTranscription(nn.Module):
    """Per-gene transcription rate regulated by transcription factors.

    For each target gene g, learns:
        alpha_g(cell) = softplus( sum_k(w_gk * TF_k_expr) + b_g )

    where w_gk is only learned for known TF-target pairs (masked by tf_target_mask).

    Parameters
    ----------
    n_genes
        Number of target genes.
    n_tfs
        Number of transcription factors.
    tf_target_mask
        Binary mask [n_genes, n_tfs] indicating known TF-target regulatory pairs.
    init_weights
        Optional initial weights for TF-target pairs [n_genes, n_tfs].
    basal_alpha_init
        Optional initial basal transcription rates [n_genes] (from VeloVI's alpha_unconstr).
    """

    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        tf_target_mask: torch.Tensor,
        init_weights: Optional[torch.Tensor] = None,
        basal_alpha_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_tfs = n_tfs

        # Static binary mask: only learn weights where known regulatory relationships exist
        self.register_buffer("tf_mask", tf_target_mask.float())

        # Learnable TF weights (masked during forward)
        if init_weights is not None:
            self.w_raw = nn.Parameter(init_weights.float())
        else:
            self.w_raw = nn.Parameter(torch.zeros(n_genes, n_tfs))

        # Basal transcription rate (warm-started from VeloVI's alpha_unconstr)
        if basal_alpha_init is not None:
            self.b_unconstr = nn.Parameter(basal_alpha_init.float())
        else:
            self.b_unconstr = nn.Parameter(torch.zeros(n_genes))

    def forward(self, tf_expression: torch.Tensor) -> torch.Tensor:
        """Compute cell-specific transcription rates.

        Parameters
        ----------
        tf_expression
            Expression of TF genes [batch, n_tfs].

        Returns
        -------
        alpha
            Cell-specific transcription rates [batch, n_genes].
        """
        # Apply mask to zero out non-regulatory pairs
        w = self.w_raw * self.tf_mask  # [n_genes, n_tfs]

        # Compute weighted TF contribution per gene
        # [batch, n_tfs] @ [n_tfs, n_genes] -> [batch, n_genes]
        weighted_tf = torch.matmul(tf_expression, w.T)

        # Add basal rate and apply softplus for positivity
        alpha = F.softplus(weighted_tf + self.b_unconstr)

        return torch.clamp(alpha, 0, 50)

    def get_weights(self) -> torch.Tensor:
        """Return the masked weights for interpretation."""
        with torch.no_grad():
            return (self.w_raw * self.tf_mask).detach().cpu()


def build_tf_mask(
    adata: AnnData,
    databases: List[str] = None,
    data_dir: Optional[str] = None,
    max_n_tf: int = 99,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build TF-target mask from ENCODE and ChEA databases.

    Adapted from TFvelo's get_TFs() but simplified to only build the binary mask
    and correlation-based initial weights.

    Parameters
    ----------
    adata
        AnnData with gene expression data. Must have spliced expression accessible.
    databases
        List of databases to use. Options: 'ENCODE', 'ChEA'. Default: ['ENCODE', 'ChEA'].
    data_dir
        Directory containing TF database files. Default: TFvelo data directory.
    max_n_tf
        Maximum number of TFs per gene.

    Returns
    -------
    tf_target_mask
        Binary mask [n_genes, n_tfs] indicating known TF-target pairs.
    tf_indices
        Indices of TF genes in adata.var_names.
    tf_names
        Names of TF genes found in the data.
    init_weights
        Correlation-based initial weights [n_genes, n_tfs].
    """
    if databases is None:
        databases = ["ENCODE", "ChEA"]

    if data_dir is None:
        # Default to TFvelo's root directory (ENCODE/ and ChEA/ are at this level)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent = os.path.dirname(base)
        data_dir = os.path.join(parent, "TFvelo")

    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    # Build case-insensitive lookup: uppercase -> actual gene name in adata
    gene_upper_to_actual = {}
    gene_actual_to_idx = {}
    for i, g in enumerate(gene_names):
        gene_upper_to_actual[g.upper()] = g
        gene_actual_to_idx[g] = i

    def _resolve_gene(name):
        """Resolve a gene name to the adata gene name (case-insensitive)."""
        if name in gene_actual_to_idx:
            return name
        upper = name.upper()
        if upper in gene_upper_to_actual:
            return gene_upper_to_actual[upper]
        return None

    # Collect all TF-target pairs from databases
    tf_target_pairs = {}  # {target_gene_idx: {tf_actual_name: count}}

    if "ENCODE" in databases:
        encode_dir = os.path.join(data_dir, "ENCODE", "processed")
        if os.path.exists(encode_dir):
            for tf_file in os.listdir(encode_dir):
                tf_db_name = tf_file.replace(".txt", "").strip()
                tf_actual = _resolve_gene(tf_db_name)
                if tf_actual is None:
                    continue
                filepath = os.path.join(encode_dir, tf_file)
                try:
                    with open(filepath, "r") as f:
                        targets = [line.strip() for line in f.readlines()]
                except Exception:
                    continue
                for target_db in targets:
                    target_actual = _resolve_gene(target_db)
                    if target_actual is not None and target_actual != tf_actual:
                        target_idx = gene_actual_to_idx[target_actual]
                        if target_idx not in tf_target_pairs:
                            tf_target_pairs[target_idx] = {}
                        if tf_actual not in tf_target_pairs[target_idx]:
                            tf_target_pairs[target_idx][tf_actual] = 0
                        tf_target_pairs[target_idx][tf_actual] += 1
        else:
            print(f"  WARNING: ENCODE directory not found: {encode_dir}")

    if "ChEA" in databases:
        chea_file = os.path.join(data_dir, "ChEA", "ChEA_2016.txt")
        if os.path.exists(chea_file):
            with open(chea_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    # First field: "TF_NAME PMID Method CellType Species"
                    tf_db_name = parts[0].split(" ")[0].split("_")[0].strip()
                    tf_actual = _resolve_gene(tf_db_name)
                    if tf_actual is None:
                        continue
                    # Targets start from column index 2
                    targets = parts[2:]
                    for target_db in targets:
                        target_db = target_db.strip()
                        if not target_db:
                            continue
                        target_actual = _resolve_gene(target_db)
                        if target_actual is not None and target_actual != tf_actual:
                            target_idx = gene_actual_to_idx[target_actual]
                            if target_idx not in tf_target_pairs:
                                tf_target_pairs[target_idx] = {}
                            if tf_actual not in tf_target_pairs[target_idx]:
                                tf_target_pairs[target_idx][tf_actual] = 0
                            tf_target_pairs[target_idx][tf_actual] += 1
        else:
            print(f"  WARNING: ChEA file not found: {chea_file}")

    # Identify all unique TFs found
    all_tfs = set()
    for target_idx, tfs in tf_target_pairs.items():
        for tf_name in tfs:
            all_tfs.add(tf_name)

    tf_names = sorted(list(all_tfs))
    n_tfs = len(tf_names)

    if n_tfs == 0:
        raise ValueError(
            f"No transcription factors found in the data from databases {databases}. "
            f"Check that data_dir={data_dir} contains the database files and that "
            f"TF names overlap with adata.var_names."
        )

    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}
    tf_indices = np.array([gene_actual_to_idx[tf] for tf in tf_names])

    # Build binary mask
    tf_target_mask = np.zeros((n_genes, n_tfs), dtype=np.float32)
    for target_idx, tfs in tf_target_pairs.items():
        tf_list = sorted(tfs.keys(), key=lambda x: tfs[x], reverse=True)[:max_n_tf]
        for tf_name in tf_list:
            tf_col = tf_name_to_idx[tf_name]
            tf_target_mask[target_idx, tf_col] = 1.0

    # Compute correlation-based initial weights
    init_weights = np.zeros((n_genes, n_tfs), dtype=np.float32)

    # Get expression data for correlation
    if "Ms" in adata.layers:
        expr_data = adata.layers["Ms"]
    elif "spliced" in adata.layers:
        expr_data = adata.layers["spliced"]
    else:
        expr_data = adata.X

    if scipy.sparse.issparse(expr_data):
        expr_data = expr_data.toarray()

    for target_idx in tf_target_pairs:
        target_expr = np.ravel(expr_data[:, target_idx])
        for tf_name in tf_target_pairs[target_idx]:
            tf_col = tf_name_to_idx[tf_name]
            if tf_target_mask[target_idx, tf_col] == 0:
                continue
            tf_gene_idx = gene_actual_to_idx[tf_name]
            tf_expr = np.ravel(expr_data[:, tf_gene_idx])

            # Filter for cells with sufficient expression
            valid = (tf_expr > 0.1) & (target_expr > 0.1)
            if valid.sum() < 2:
                init_weights[target_idx, tf_col] = 0.0
            else:
                corr, _ = scipy.stats.spearmanr(target_expr[valid], tf_expr[valid])
                init_weights[target_idx, tf_col] = corr if not np.isnan(corr) else 0.0

    # Scale initial weights to a reasonable range for softplus input
    init_weights = init_weights * 0.1

    return tf_target_mask, tf_indices, tf_names, init_weights
