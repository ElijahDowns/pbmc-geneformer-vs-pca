"""Standard Scanpy baseline: normalise, log1p, 2,000 highly variable genes, scale, 50-component PCA."""

import scanpy as sc

from pbmc_fm import SEED

N_TOP_GENES = 2000
N_COMPONENTS = 50


def pca_embedding(adata):
    """Return the (n_cells, 50) PCA embedding of raw counts, computed on a copy of `adata`."""
    adata = adata.copy()
    # Make cells comparable regardless of sequencing depth, then compress the long tail of counts.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    adata = adata[:, adata.var["highly_variable"]].copy()
    # Centring each gene makes the matrix dense anyway; 2,638 x 2,000 values fit easily in memory.
    adata.X = adata.X.toarray()

    # Unit variance per gene so a few very highly expressed genes do not dominate the PCs.
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_COMPONENTS, svd_solver="arpack", random_state=SEED)
    return adata.obsm["X_pca"]
