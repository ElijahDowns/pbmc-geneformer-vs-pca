"""Load PBMC3k raw counts, attach the published cell-type labels, and record basic QC metrics."""

from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

# Cells in Scanpy's processed PBMC3k, i.e. the cells that passed the original tutorial's QC.
EXPECTED_N_CELLS = 2638


def load_raw_counts(data_dir):
    """Download (first call only) and return the raw PBMC3k UMI counts: 2,700 cells x 32,738 genes."""
    sc.settings.datasetdir = Path(data_dir)
    raw = sc.datasets.pbmc3k()
    # Some gene symbols appear twice; Ensembl IDs are unique and stay in var["gene_ids"].
    raw.var_names_make_unique()
    return raw


def load_labels(data_dir):
    """Return the cell-type label of every cell in Scanpy's processed PBMC3k, indexed by barcode."""
    sc.settings.datasetdir = Path(data_dir)
    processed = sc.datasets.pbmc3k_processed()
    return processed.obs["louvain"].copy()


def attach_labels(raw, labels):
    """Return a copy of `raw` restricted to the labelled barcodes (in label order) with obs["cell_type"]."""
    missing = labels.index.difference(raw.obs_names)
    assert len(missing) == 0, f"{len(missing)} labelled barcodes are not in the raw data"

    adata = raw[labels.index].copy()
    assert list(adata.obs_names) == list(labels.index)
    adata.obs["cell_type"] = labels.values
    return adata


def add_qc_metrics(adata):
    """Return a copy without genes seen in fewer than 3 cells, with per-cell QC metrics in obs.

    Cells are not filtered here: the labelled cells already passed the original tutorial's QC.
    """
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # Adds n_genes_by_counts, total_counts and pct_counts_mt to obs.
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return adata


def is_integer_counts(matrix):
    """True if every stored value is a non-negative whole number, as raw UMI counts must be."""
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    return bool(np.all(values >= 0) and np.all(values == np.round(values)))


def prepare_labelled_pbmc3k(data_dir):
    """Raw counts for the 2,638 labelled PBMC3k cells, with obs["cell_type"] and QC metrics."""
    raw = load_raw_counts(data_dir)
    labels = load_labels(data_dir)
    adata = attach_labels(raw, labels)
    adata = add_qc_metrics(adata)

    assert adata.n_obs == EXPECTED_N_CELLS, f"expected {EXPECTED_N_CELLS} cells, got {adata.n_obs}"
    # Geneformer tokenises cells by ranking raw counts, so normalised values would be wrong input.
    assert is_integer_counts(adata.X), "X must hold raw integer counts"
    return adata
