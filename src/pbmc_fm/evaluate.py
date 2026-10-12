"""Metrics applied identically to every embedding: kNN label accuracy, best Leiden ARI/NMI, silhouette."""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from pbmc_fm import SEED

N_NEIGHBORS = 15
N_FOLDS = 5
# 0.1, 0.2, ..., 2.0, built from integers so values are exact (no 0.30000000000000004).
LEIDEN_RESOLUTIONS = [round(0.1 * step, 1) for step in range(1, 21)]

# Markers the Scanpy PBMC3k tutorial uses to name these cell types, plus genes shared across
# related types (CD3E: all T cells; GZMA/CCL5: cytotoxic NK and CD8 T cells).
MARKER_GENES = [
    "CD3E", "IL7R", "CCR7", "CD8A", "CD8B", "GZMA", "CCL5", "NKG7", "GNLY",
    "MS4A1", "CD79A", "CD14", "LYZ", "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP",
]


def knn_predictions(embedding, labels):
    """Predict each cell's type from its 15 nearest neighbours while that cell is held out.

    5-fold stratified cross-validation, so every cell is predicted exactly once by a model
    that never saw it, and each fold keeps the overall cell-type proportions.
    """
    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(classifier, embedding, labels, cv=folds)


def neighbour_graph(embedding):
    """AnnData holding the embedding and its 15-nearest-neighbour graph (used by Leiden and UMAP)."""
    # AnnData expects string cell names; row order is the embedding's (i.e. the labelled data's).
    obs = pd.DataFrame(index=[str(i) for i in range(len(embedding))])
    adata = ad.AnnData(obs=obs, obsm={"X_emb": np.asarray(embedding)})
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=N_NEIGHBORS, random_state=SEED)
    return adata


def run_leiden(adata, resolution):
    """Leiden clusters of the cells in a neighbour graph, with the settings used throughout."""
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=SEED,
        key_added="leiden",
    )
    return adata.obs["leiden"].to_numpy()


def leiden_clusters(embedding, resolution):
    """Leiden cluster of each cell at one resolution (same graph and settings as the sweep)."""
    return run_leiden(neighbour_graph(embedding), resolution)


def leiden_sweep(embedding, labels):
    """ARI, NMI and cluster count of a Leiden clustering of the embedding at each resolution."""
    adata = neighbour_graph(embedding)
    rows = []
    for resolution in LEIDEN_RESOLUTIONS:
        clusters = run_leiden(adata, resolution)
        rows.append(
            {
                "resolution": resolution,
                "n_clusters": len(set(clusters)),
                "ari": adjusted_rand_score(labels, clusters),
                "nmi": normalized_mutual_info_score(labels, clusters),
            }
        )
    return pd.DataFrame(rows)


def evaluate_embedding(embedding, labels):
    """Score one embedding against the ground-truth labels.

    Returns the headline metrics (one row of results/metrics.csv), the full Leiden sweep,
    and each cell's held-out kNN prediction (for the per-cell-type table).
    """
    labels = np.asarray(labels)
    predictions = knn_predictions(embedding, labels)
    sweep = leiden_sweep(embedding, labels)
    # Best score over the sweep, so neither embedding is penalised by one arbitrary resolution.
    best_ari = sweep.loc[sweep["ari"].idxmax()]
    best_nmi = sweep.loc[sweep["nmi"].idxmax()]

    metrics = {
        "knn_accuracy": float(np.mean(predictions == labels)),
        "ari": float(best_ari["ari"]),
        "ari_resolution": float(best_ari["resolution"]),
        "ari_n_clusters": int(best_ari["n_clusters"]),
        "nmi": float(best_nmi["nmi"]),
        "nmi_resolution": float(best_nmi["resolution"]),
        "silhouette_cosine": float(silhouette_score(embedding, labels, metric="cosine")),
    }
    return metrics, sweep, predictions


def confusion_table(predictions, labels):
    """Row-normalised confusion matrix: for each true cell type, the share of its cells given each label."""
    true = pd.Series(np.asarray(labels), name="cell_type")
    predicted = pd.Series(np.asarray(predictions), name="predicted")
    return pd.crosstab(true, predicted, normalize="index")


def per_type_summary(predictions, labels):
    """Per cell type: number of cells, held-out kNN accuracy, and the wrong label it most often gets."""
    labels = np.asarray(labels)
    confusion = confusion_table(predictions, labels)
    rows = []
    for cell_type in confusion.index:
        shares = confusion.loc[cell_type]
        wrong = shares.drop(cell_type, errors="ignore")
        rows.append(
            {
                "cell_type": cell_type,
                "n_cells": int(np.sum(labels == cell_type)),
                "accuracy": float(shares.get(cell_type, 0.0)),
                "most_confused_with": wrong.idxmax() if wrong.max() > 0 else "",
                "confused_share": float(wrong.max()),
            }
        )
    return pd.DataFrame(rows).set_index("cell_type")


def cluster_composition(adata, clusters):
    """Per Leiden cluster: cells of each type, median genes detected, and two CD4 T subset markers.

    Shows which cell types a clustering merges or splits. CCR7 marks naive and S100A4 memory
    CD4 T cells, so they show whether a split of the CD4 T cells follows that biology.
    """
    table = pd.crosstab(clusters, adata.obs["cell_type"].to_numpy(), rownames=["cluster"], colnames=[None])
    table["median_genes_detected"] = adata.obs["n_genes_by_counts"].groupby(clusters).median()
    for gene in ["CCR7", "S100A4"]:
        counts = adata[:, gene].X
        if sp.issparse(counts):
            counts = counts.toarray()
        table[f"{gene}_detected"] = pd.Series(counts.ravel() > 0).groupby(clusters).mean()
    return table


def marker_detection(adata, genes=MARKER_GENES):
    """Share of each cell type's cells with at least one UMI of each marker gene (from raw counts)."""
    counts = adata[:, genes].X
    if sp.issparse(counts):
        counts = counts.toarray()
    detected = pd.DataFrame(counts > 0, index=adata.obs_names, columns=genes)
    return detected.groupby(adata.obs["cell_type"].to_numpy()).mean()
