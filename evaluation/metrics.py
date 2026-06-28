"""Comprehensive evaluation metrics for TARVI.

Metrics implemented:
- Embedding quality: silhouette, ARI, local/global preservation
- Velocity quality: ICVCoh, CBDir (fixed), velocity confidence, coherence length
- Negative control robustness: STS (self-transition score), NTE (normalized transition entropy)
- Pseudotime: distance correlation, Spearman, local consistency, Kendall tau, CTO
- ODE fit: gene-level velocity R², state assignment entropy/purity
- Directional: uncertainty, transition probability score

CBDir uses cell displacement vectors (z_j - z_i), matching the scVelo / benchmarking
  paper definition, and supports directed source->target transitions (falling back to
  symmetric all-pairs when transitions=None).
ICCoH uses a kNN-based local approach (benchmark definition).
STS / NTE: negative-control robustness from the 2026 benchmarking paper (Liu et al.),
  computed on the velocity transition graph.
CTO: cluster temporal ordering accuracy using ground-truth discrete time labels.
Velocity NaNs are sanitized to 0.0 before metric computation so results are never null.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
import logging

logger = logging.getLogger(__name__)


def compute_all_metrics(
    adata,
    velocity_key="velocity",
    latent_time_key="latent_time",
    latent_key="X_tarvi",
    cluster_key="clusters",
    umap_key="X_umap",
    n_neighbors=15,
    state_probs=None,
    s_hat=None,
    u_hat=None,
    directional_uncertainty=None,
    gt_time_key=None,
    cbdir_transitions=None,
):
    """Compute all evaluation metrics from an AnnData with velocity results.

    Parameters
    ----------
    adata
        AnnData with velocity, latent_time, and latent representation.
    velocity_key
        Key in adata.layers for velocity.
    latent_time_key
        Key in adata.obs for latent time.
    latent_key
        Key in adata.obsm for latent representation.
    cluster_key
        Key in adata.obs for cluster labels.
    umap_key
        Key in adata.obsm for 2D embedding.
    n_neighbors
        Number of neighbors for kNN-based metrics.
    state_probs
        Optional state assignment probabilities [N, G, 4].
    s_hat, u_hat
        Optional ODE-predicted spliced/unspliced [N, G].
    directional_uncertainty
        Optional per-cell uncertainty [N].
    gt_time_key
        Optional key in adata.obs for ground-truth continuous time (for CTO/TSC).

    Returns
    -------
    dict of metric name -> value
    """
    import scipy.sparse

    metrics = {}

    velocity = adata.layers[velocity_key]
    if scipy.sparse.issparse(velocity):
        velocity = velocity.toarray()
    velocity = np.array(velocity, dtype=float)
    # Replace NaN/inf from failed ODE fits (e.g. scVelo_dyn) so metrics are never null
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    latent_time = np.array(adata.obs[latent_time_key], dtype=float)
    z_high = np.array(adata.obsm[latent_key], dtype=float)

    labels = np.array(adata.obs[cluster_key])
    unique_labels = np.unique(labels)
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[l] for l in labels])
    n_clusters = len(unique_labels)
    unique_labels_int = np.arange(n_clusters)

    umap_emb = None
    if umap_key in adata.obsm:
        umap_emb = np.array(adata.obsm[umap_key], dtype=float)

    if "Ms" in adata.layers:
        s_obs = adata.layers["Ms"]
    elif "spliced" in adata.layers:
        s_obs = adata.layers["spliced"]
    else:
        s_obs = adata.X
    if scipy.sparse.issparse(s_obs):
        s_obs = s_obs.toarray()
    s_obs = np.array(s_obs, dtype=float)

    if "Mu" in adata.layers:
        u_obs = adata.layers["Mu"]
    elif "unspliced" in adata.layers:
        u_obs = adata.layers["unspliced"]
    else:
        u_obs = None
    if u_obs is not None and scipy.sparse.issparse(u_obs):
        u_obs = u_obs.toarray()
    if u_obs is not None:
        u_obs = np.array(u_obs, dtype=float)

    z_2d = umap_emb if umap_emb is not None else z_high[:, :2]

    # ── Embedding quality ─────────────────────────────────────────────────────
    logger.info("Computing embedding quality metrics...")
    metrics["local_preservation"] = _local_preservation(z_high, z_2d, k=n_neighbors)
    metrics["distance_correlation"] = _distance_correlation(z_high, z_2d, n_pairs=2000)
    metrics["global_preservation"] = _global_preservation(z_high, z_2d, n_pairs=2000)

    if n_clusters > 1:
        metrics["silhouette_2d"] = float(silhouette_score(z_2d, labels_int))
        metrics["silhouette_high"] = float(silhouette_score(z_high, labels_int))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(z_2d)
        metrics["ari_2d"] = float(adjusted_rand_score(labels_int, km.labels_))
    else:
        metrics["silhouette_2d"] = 0.0
        metrics["silhouette_high"] = 0.0
        metrics["ari_2d"] = 0.0

    # ── Velocity quality ──────────────────────────────────────────────────────
    logger.info("Computing velocity quality metrics...")
    metrics.update(_in_cluster_coherence(
        velocity, z_high, labels_int, unique_labels_int, n_neighbors
    ))
    metrics.update(_cross_boundary_direction(
        velocity,
        z_high,
        labels_int,
        unique_labels_int,
        n_neighbors,
        labels=labels,
        transitions=cbdir_transitions,
    ))
    metrics["velocity_confidence_mean"] = _velocity_confidence(
        velocity, z_high, n_neighbors
    )
    metrics["velocity_coherence_length"] = _velocity_coherence_length(
        velocity, z_high, n_neighbors
    )
    metrics["velocity_magnitude_corr"] = _velocity_magnitude_correlation(
        velocity, z_high, s_obs, n_neighbors
    )

    # ── Negative control robustness (STS / NTE) ───────────────────────────────
    logger.info("Computing negative control robustness metrics...")
    sts, nte = _self_transition_and_entropy(adata)
    metrics["self_transition_score"] = sts
    metrics["normalized_transition_entropy"] = nte

    # ── Pseudotime ────────────────────────────────────────────────────────────
    logger.info("Computing pseudotime metrics...")
    metrics["pseudotime_distance_corr_2d"] = _pseudotime_distance_corr(latent_time, z_2d)
    metrics["pseudotime_distance_corr_high"] = _pseudotime_distance_corr(latent_time, z_high)
    metrics["pseudotime_spearman_2d"] = _pseudotime_spearman(latent_time, z_2d)
    metrics["pseudotime_spearman_high"] = _pseudotime_spearman(latent_time, z_high)
    metrics["pseudotime_local_consistency"] = _pseudotime_local_consistency(
        latent_time, z_high, n_neighbors
    )
    metrics["pseudotime_kendall_tau"] = _pseudotime_kendall_tau(
        latent_time, z_high, n_neighbors
    )
    metrics["root_cell_accuracy"] = _root_cell_accuracy(
        latent_time, labels_int, unique_labels, adata, cluster_key
    )

    # CTO: cluster temporal ordering (ground-truth discrete or continuous time)
    if gt_time_key is not None and gt_time_key in adata.obs:
        gt_time = np.array(adata.obs[gt_time_key], dtype=float)
        metrics["cluster_temporal_ordering"] = _cluster_temporal_ordering(
            latent_time, gt_time, labels_int, unique_labels_int
        )
        metrics["temporal_spearman_corr"] = _gt_temporal_spearman(latent_time, gt_time)

    # ── ODE / kinetic metrics ─────────────────────────────────────────────────
    logger.info("Computing ODE-specific metrics...")
    if s_hat is not None and u_hat is not None and u_obs is not None:
        gene_r2_s, gene_r2_u, gene_r2_mean = _gene_velocity_r2(s_hat, u_hat, s_obs, u_obs)
        metrics["gene_velocity_r2_spliced"] = gene_r2_s
        metrics["gene_velocity_r2_unspliced"] = gene_r2_u
        metrics["gene_velocity_r2_mean"] = gene_r2_mean

    if state_probs is not None:
        state_probs_np = np.array(state_probs)
        metrics["state_assignment_entropy"] = _state_assignment_entropy(state_probs_np)
        metrics["state_assignment_purity"] = _state_assignment_purity(state_probs_np)

    if directional_uncertainty is not None:
        unc = np.array(directional_uncertainty)
        metrics["directional_uncertainty_mean"] = float(unc.mean())
        metrics["directional_uncertainty_std"] = float(unc.std())

    # ── 2D-embedding-specific metrics ────────────────────────────────────────
    if umap_emb is not None:
        logger.info("Computing 2D-embedding-specific metrics...")
        metrics["umap_local_preservation"] = _local_preservation(z_high, umap_emb, k=n_neighbors)
        metrics["umap_distance_correlation"] = _distance_correlation(z_high, umap_emb, n_pairs=2000)
        if n_clusters > 1:
            metrics["umap_silhouette_2d"] = float(silhouette_score(umap_emb, labels_int))
            km_umap = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(umap_emb)
            metrics["umap_ari_2d"] = float(adjusted_rand_score(labels_int, km_umap.labels_))
        metrics["umap_pseudotime_distance_corr_2d"] = _pseudotime_distance_corr(latent_time, umap_emb)
        metrics["umap_pseudotime_spearman_2d"] = _pseudotime_spearman(latent_time, umap_emb)
        metrics["umap_pseudotime_local_consistency"] = _pseudotime_local_consistency(
            latent_time, umap_emb, n_neighbors
        )

    # ── Transition probability asymmetry ─────────────────────────────────────
    metrics["transition_probability_score"] = _transition_probability_score(
        velocity, z_high, labels_int, unique_labels_int, n_neighbors
    )

    logger.info(f"Computed {len(metrics)} metrics total")
    return metrics


# ── Embedding quality ──────────────────────────────────────────────────────────

def _local_preservation(z_high, z_2d, k=15):
    nn_high = NearestNeighbors(n_neighbors=k).fit(z_high)
    nn_2d = NearestNeighbors(n_neighbors=k).fit(z_2d)
    _, idx_high = nn_high.kneighbors(z_high)
    _, idx_2d = nn_2d.kneighbors(z_2d)
    scores = []
    for i in range(len(z_high)):
        s_h = set(idx_high[i]); s_2 = set(idx_2d[i])
        union = len(s_h | s_2)
        scores.append(len(s_h & s_2) / union if union > 0 else 0)
    return float(np.mean(scores))


def _distance_correlation(z_high, z_2d, n_pairs=2000):
    n = len(z_high)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    ii = rng.randint(0, n, n_pairs); jj = rng.randint(0, n, n_pairs)
    mask = ii != jj; ii, jj = ii[mask], jj[mask]
    d_h = np.sqrt(((z_high[ii] - z_high[jj]) ** 2).sum(axis=1))
    d_2 = np.sqrt(((z_2d[ii] - z_2d[jj]) ** 2).sum(axis=1))
    corr, _ = pearsonr(d_h, d_2)
    return float(corr) if not np.isnan(corr) else 0.0


def _global_preservation(z_high, z_2d, n_pairs=2000):
    n = len(z_high)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    ii = rng.randint(0, n, n_pairs); jj = rng.randint(0, n, n_pairs)
    mask = ii != jj; ii, jj = ii[mask], jj[mask]
    d_h = np.sqrt(((z_high[ii] - z_high[jj]) ** 2).sum(axis=1))
    d_2 = np.sqrt(((z_2d[ii] - z_2d[jj]) ** 2).sum(axis=1))
    corr, _ = spearmanr(d_h, d_2)
    return float(corr) if not np.isnan(corr) else 0.0


# ── Velocity quality ───────────────────────────────────────────────────────────

def _in_cluster_coherence(velocity, z_high, labels_int, unique_labels, k=15):
    """Benchmark-style in-cluster coherence over same-cluster kNN neighbors."""
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    vel_norm = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)
    cell_scores = []
    for cell_idx in range(len(velocity)):
        same_cluster = nn_idx[cell_idx][labels_int[nn_idx[cell_idx]] == labels_int[cell_idx]]
        same_cluster = same_cluster[same_cluster != cell_idx]
        if len(same_cluster) == 0:
            continue
        scores = (vel_norm[cell_idx] * vel_norm[same_cluster]).sum(axis=1)
        cell_scores.append(float(scores.mean()))
    return {"iccoh_overall": float(np.mean(cell_scores)) if cell_scores else 0.0}


def _cross_boundary_direction(
    velocity,
    z_high,
    labels_int,
    unique_labels,
    k=15,
    labels=None,
    transitions=None,
):
    """Cross-boundary direction correctness.

    When explicit source-target transitions are supplied, this follows the
    benchmark definition: compare each boundary source cell velocity with the
    displacement from that source cell to neighboring target cells. If no
    transitions are supplied, the legacy all-label-pairs behavior is retained
    for backward compatibility.
    """
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    vel_norm = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)

    all_scores = []
    if transitions is not None and labels is not None:
        label_values = np.array([str(x) for x in labels])
        for source, target in transitions:
            source = str(source)
            target = str(target)
            source_mask = label_values == source
            target_mask = label_values == target
            if source_mask.sum() == 0 or target_mask.sum() == 0:
                continue

            for c_idx in np.where(source_mask)[0]:
                target_neighbors = nn_idx[c_idx][target_mask[nn_idx[c_idx]]]
                if len(target_neighbors) == 0:
                    continue
                target_disp = z_high[target_neighbors].mean(axis=0) - z_high[c_idx]
                target_disp /= np.linalg.norm(target_disp) + 1e-8
                all_scores.append(float((vel_norm[c_idx] * target_disp).sum()))

        return {"cbdir_overall": float(np.mean(all_scores)) if all_scores else 0.0}

    for i, label_i in enumerate(unique_labels):
        mask_i = labels_int == label_i
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue

            boundary_cells = []
            for c_idx in np.where(mask_i)[0]:
                nbr_labels = labels_int[nn_idx[c_idx]]
                if (nbr_labels == label_j).any():
                    boundary_cells.append(c_idx)

            if len(boundary_cells) < 3:
                continue

            for c_idx in boundary_cells:
                j_neighbors = nn_idx[c_idx][labels_int[nn_idx[c_idx]] == label_j]
                if len(j_neighbors) == 0:
                    continue
                mean_nbr_vel = vel_norm[j_neighbors].mean(axis=0)
                mean_nbr_vel_n = mean_nbr_vel / (np.linalg.norm(mean_nbr_vel) + 1e-8)
                all_scores.append(float((vel_norm[c_idx] * mean_nbr_vel_n).sum()))

    return {"cbdir_overall": float(np.mean(all_scores)) if all_scores else 0.0}


def _velocity_confidence(velocity, z_high, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    vel_norm = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)
    confidence = []
    for i in range(len(velocity)):
        nbr_vel = vel_norm[nn_idx[i]]
        confidence.append((vel_norm[i] * nbr_vel).sum(axis=1).mean())
    return float(np.mean(confidence))


def _velocity_coherence_length(velocity, z_high, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    distances, nn_idx = nn_model.kneighbors(z_high)
    vel_norm = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)
    lengths = []
    for i in range(len(velocity)):
        cos_sims = (vel_norm[i] * vel_norm[nn_idx[i]]).sum(axis=1)
        coherent = cos_sims > 0.5
        lengths.append(distances[i][coherent].max() if coherent.any() else 0.0)
    return float(np.mean(lengths))


def _velocity_magnitude_correlation(velocity, z_high, s_obs, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    vel_mag = np.linalg.norm(velocity, axis=1)
    expr_change = np.array([
        np.linalg.norm(s_obs[i] - s_obs[nn_idx[i]].mean(axis=0))
        for i in range(len(s_obs))
    ])
    if vel_mag.std() < 1e-8 or expr_change.std() < 1e-8:
        return 0.0
    corr, _ = spearmanr(vel_mag, expr_change)
    return float(corr) if not np.isnan(corr) else 0.0


# ── Negative control robustness ────────────────────────────────────────────────

def _self_transition_and_entropy(adata):
    """STS and NTE from the velocity transition graph (Liu et al. 2026).

    STS (Self-Transition Score): mean diagonal of the velocity transition matrix.
      High STS → cells mostly transition to themselves → robust in steady state.

    NTE (Normalized Transition Entropy): mean row-entropy of transition matrix
      normalized by log(n_cells). High NTE → diffuse, near-uniform transitions
      → method does not over-commit to spurious trajectories.

    Both require adata.uns['velocity_graph'] to be set.
    """
    import scipy.sparse as sp

    if "velocity_graph" not in adata.uns:
        return float("nan"), float("nan")

    T = adata.uns["velocity_graph"]
    if sp.issparse(T):
        T = T.toarray()
    T = np.array(T, dtype=float)
    n = T.shape[0]

    # velocity_graph has no self-loops (diagonal = 0 by scVelo convention).
    # Add self-connection weight = 1.0 so that cells with weak directional
    # signal retain most probability on themselves (STS → 1), while cells
    # with strong outgoing velocity (many large off-diagonal entries) get
    # lower STS. This matches the spirit of Liu et al. 2026.
    T_with_self = T.copy()
    np.fill_diagonal(T_with_self, 1.0)

    # Row-normalise to get a proper transition probability matrix
    row_sums = T_with_self.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T_prob = T_with_self / row_sums

    # STS: mean self-transition probability
    sts = float(np.diag(T_prob).mean())

    # NTE: mean row-entropy of T_prob, normalised by log(n)
    T_safe = np.clip(T_prob, 1e-12, 1.0)
    entropy = -(T_prob * np.log(T_safe)).sum(axis=1)
    max_entropy = np.log(n) if n > 1 else 1.0
    nte = float((entropy / max_entropy).mean())

    return sts, nte


# ── Pseudotime ─────────────────────────────────────────────────────────────────

def _pseudotime_distance_corr(latent_time, embedding):
    n = len(latent_time)
    n_pairs = min(2000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    ii = rng.randint(0, n, n_pairs); jj = rng.randint(0, n, n_pairs)
    mask = ii != jj; ii, jj = ii[mask], jj[mask]
    d_emb = np.sqrt(((embedding[ii] - embedding[jj]) ** 2).sum(axis=1))
    d_t = np.abs(latent_time[ii] - latent_time[jj])
    corr, _ = pearsonr(d_emb, d_t)
    return float(corr) if not np.isnan(corr) else 0.0


def _pseudotime_spearman(latent_time, embedding):
    n = len(latent_time)
    n_pairs = min(2000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    ii = rng.randint(0, n, n_pairs); jj = rng.randint(0, n, n_pairs)
    mask = ii != jj; ii, jj = ii[mask], jj[mask]
    d_emb = np.sqrt(((embedding[ii] - embedding[jj]) ** 2).sum(axis=1))
    d_t = np.abs(latent_time[ii] - latent_time[jj])
    corr, _ = spearmanr(d_emb, d_t)
    return float(corr) if not np.isnan(corr) else 0.0


def _pseudotime_local_consistency(latent_time, embedding, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(embedding)
    _, nn_idx = nn_model.kneighbors(embedding)
    time_range = latent_time.max() - latent_time.min()
    if time_range < 1e-8:
        return 0.0
    threshold = 0.1 * time_range
    consistent = total = 0
    for i in range(len(latent_time)):
        for j_idx in nn_idx[i]:
            if abs(latent_time[i] - latent_time[j_idx]) < threshold:
                consistent += 1
            total += 1
    return consistent / total if total > 0 else 0.0


def _pseudotime_kendall_tau(latent_time, z_high, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(latent_time), min(500, len(latent_time)), replace=False)
    taus = []
    for i in sample_idx:
        nbrs = nn_idx[i]
        t_nbr = latent_time[nbrs]
        d_nbr = np.linalg.norm(z_high[nbrs] - z_high[i], axis=1)
        if t_nbr.std() < 1e-8 or d_nbr.std() < 1e-8:
            continue
        tau, _ = kendalltau(t_nbr, d_nbr)
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


def _root_cell_accuracy(latent_time, labels_int, unique_labels, adata, cluster_key):
    if cluster_key not in adata.obs:
        return 0.0
    labels = adata.obs[cluster_key].values
    cluster_mean_time = {l: latent_time[labels == l].mean() for l in np.unique(labels)}
    sorted_clusters = sorted(cluster_mean_time.items(), key=lambda x: x[1])
    n_root = max(1, int(0.1 * len(latent_time)))
    root_labels = labels[np.argsort(latent_time)[:n_root]]
    early = {sorted_clusters[0][0]}
    if len(sorted_clusters) > 1:
        early.add(sorted_clusters[1][0])
    return float(np.mean([l in early for l in root_labels]))


def _cluster_temporal_ordering(latent_time, gt_time, labels_int, unique_labels):
    """CTO: fraction of cluster pairs where inferred ordering matches GT ordering.

    For each pair of clusters (A, B), compare mean(inferred_time_A) < mean(inferred_time_B)
    with mean(gt_time_A) < mean(gt_time_B). CTO = fraction of correctly ordered pairs.
    """
    cluster_inferred = {l: latent_time[labels_int == l].mean() for l in unique_labels}
    cluster_gt = {l: gt_time[labels_int == l].mean() for l in unique_labels}

    correct = total = 0
    for i in unique_labels:
        for j in unique_labels:
            if i >= j:
                continue
            inf_order = cluster_inferred[i] < cluster_inferred[j]
            gt_order = cluster_gt[i] < cluster_gt[j]
            if inf_order == gt_order:
                correct += 1
            total += 1
    return float(correct / total) if total > 0 else 0.5


def _gt_temporal_spearman(latent_time, gt_time):
    """TSC: cell-level Spearman correlation between inferred and GT time."""
    if gt_time.std() < 1e-8 or latent_time.std() < 1e-8:
        return 0.0
    corr, _ = spearmanr(latent_time, gt_time)
    return float(abs(corr)) if not np.isnan(corr) else 0.0


# ── ODE / kinetic metrics ──────────────────────────────────────────────────────

def _gene_velocity_r2(s_hat, u_hat, s_obs, u_obs):
    r2_s, r2_u = [], []
    for g in range(s_obs.shape[1]):
        ss_res = ((s_obs[:, g] - s_hat[:, g]) ** 2).sum()
        ss_tot = ((s_obs[:, g] - s_obs[:, g].mean()) ** 2).sum()
        r2_s.append(max(-1.0, 1.0 - ss_res / (ss_tot + 1e-8)))

        ss_res_u = ((u_obs[:, g] - u_hat[:, g]) ** 2).sum()
        ss_tot_u = ((u_obs[:, g] - u_obs[:, g].mean()) ** 2).sum()
        r2_u.append(max(-1.0, 1.0 - ss_res_u / (ss_tot_u + 1e-8)))

    return float(np.mean(r2_s)), float(np.mean(r2_u)), float(np.mean(r2_s + r2_u))


def _state_assignment_entropy(state_probs):
    entropy = -(state_probs * np.log(state_probs + 1e-8)).sum(axis=-1)
    return float(entropy.mean())


def _state_assignment_purity(state_probs):
    return float(state_probs.max(axis=-1).mean())


# ── Transition probability asymmetry ──────────────────────────────────────────

def _transition_probability_score(velocity, z_high, labels_int, unique_labels, k=15):
    nn_model = NearestNeighbors(n_neighbors=k).fit(z_high)
    _, nn_idx = nn_model.kneighbors(z_high)
    vel_norm = velocity / (np.linalg.norm(velocity, axis=1, keepdims=True) + 1e-8)

    T = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(len(velocity)):
        src = labels_int[i]
        for j_idx in nn_idx[i]:
            dst = labels_int[j_idx]
            if dst != src:
                cos_sim = (vel_norm[i] * vel_norm[j_idx]).sum()
                if cos_sim > 0:
                    T[src, dst] += cos_sim

    row_sums = T.sum(axis=1, keepdims=True)
    T = T / (row_sums + 1e-8)

    scores = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            p_ij, p_ji = T[i, j], T[j, i]
            if p_ij + p_ji > 0.01:
                scores.append(abs(p_ij - p_ji) / (p_ij + p_ji))
    return float(np.mean(scores)) if scores else 0.0
