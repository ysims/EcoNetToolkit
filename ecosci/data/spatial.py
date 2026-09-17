"""Spatial partitioning utilities.

Random train/test splits on geo-referenced ecological data tend to place
spatially autocorrelated neighbours (e.g. adjacent plots) on both sides of the
split, which lets models "cheat" by memorising local conditions rather than
learning a relationship that generalises to new sites. Grouping samples into
spatial blocks first, then splitting/cross-validating on those blocks (e.g.
via GroupKFold), keeps neighbours together on one side of the split.
"""

import pandas as pd


def assign_spatial_blocks(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    n_blocks: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Cluster samples into spatially coherent blocks using KMeans on coordinates.

    Parameters
    ----------
    df : DataFrame
        Must contain `lon_col` and `lat_col`.
    lon_col, lat_col : str
        Coordinate column names.
    n_blocks : int
        Number of spatial blocks (i.e. groups) to create. Capped at the number
        of rows if there are fewer samples than blocks.
    random_state : int
        Seed for KMeans initialisation.

    Returns
    -------
    Series
        Integer block label per row, named "spatial_block", aligned to `df`'s index.
    """
    from sklearn.cluster import KMeans

    coords = df[[lon_col, lat_col]].to_numpy()
    n_blocks = max(1, min(n_blocks, len(df)))

    km = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)

    counts = pd.Series(labels).value_counts()
    min_count = int(counts.min())
    if min_count < 2:
        n_singleton = int((counts < 2).sum())
        print(
            f"Warning: {n_singleton} of {n_blocks} spatial block(s) have fewer than "
            f"2 samples (smallest: {min_count}). A CV fold built from one of these "
            f"has an undefined test-set R^2/variance and will show up as NaN in "
            f"per-fold metrics and permutation-based feature importance. Consider "
            f"lowering n_blocks (currently {n_blocks})."
        )
    elif min_count < 5:
        print(
            f"Warning: smallest spatial block has only {min_count} samples out of "
            f"{n_blocks} blocks. Per-fold metrics for that block may be noisy/unstable. "
            f"Consider lowering n_blocks if this is unintentional."
        )

    return pd.Series(labels, index=df.index, name="spatial_block")
