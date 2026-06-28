"""Utility functions for TARVI."""
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler


def get_permutation_scores(save_path: Union[str, Path] = Path("data/")) -> pd.DataFrame:
    """Get the reference permutation scores on positive and negative controls.

    Parameters
    ----------
    save_path
        path to save the csv file

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not (save_path / "permutation_scores.csv").is_file():
        URL = "https://figshare.com/ndownloader/files/36658185"
        urlretrieve(url=URL, filename=save_path / "permutation_scores.csv")

    return pd.read_csv(save_path / "permutation_scores.csv")


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata


def smooth_velocity(
    velocity: np.ndarray,
    connectivities,
    alpha: float = 0.5,
) -> np.ndarray:
    """Smooth velocity using kNN graph.

    Simple non-parametric smoothing: weighted average of cell's own velocity
    and its neighbors' velocities.

    Parameters
    ----------
    velocity
        Velocity matrix [n_cells, n_genes].
    connectivities
        Sparse connectivity matrix from scanpy/scvelo (adata.obsp['connectivities']).
    alpha
        Mixing weight. 0 = no smoothing, 1 = full neighbor average.

    Returns
    -------
    Smoothed velocity [n_cells, n_genes].
    """
    import scipy.sparse as sp

    conn = connectivities.copy()
    # Normalize rows to sum to 1
    row_sums = np.array(conn.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    conn = sp.diags(1.0 / row_sums) @ conn

    # Weighted average: (1-alpha) * own + alpha * neighbors
    smoothed = (1 - alpha) * velocity + alpha * (conn @ velocity)
    return smoothed
