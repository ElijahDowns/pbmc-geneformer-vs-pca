"""Step 4: score every embedding with the same metrics, then write the results tables and figures."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pbmc_fm import set_seeds
from pbmc_fm.evaluate import (
    cluster_composition,
    confusion_table,
    evaluate_embedding,
    leiden_clusters,
    marker_detection,
    per_type_summary,
)
from pbmc_fm.plots import plot_metrics, plot_umaps, umap_coordinates

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "pbmc3k_labelled.h5ad"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Name -> (file written by scripts 02/03, title for figures). Rows follow data/pbmc3k_labelled.h5ad.
EMBEDDINGS = {
    "pca": (RESULTS_DIR / "pca_embedding.npy", "Scanpy PCA (50 PCs)"),
    "geneformer": (RESULTS_DIR / "geneformer_embedding.npy", "Geneformer, final layer (512-d)"),
    "geneformer_layer11": (RESULTS_DIR / "geneformer_layer11_embedding.npy", "Geneformer, second-to-last layer (512-d)"),
}

set_seeds()
FIGURES_DIR.mkdir(exist_ok=True)
adata = ad.read_h5ad(INPUT_PATH)
labels = adata.obs["cell_type"].to_numpy()

metric_rows = {}
sweeps = []
per_type = pd.DataFrame({"n_cells": pd.Series(labels).value_counts()})
umaps = {}
for name, (path, title) in EMBEDDINGS.items():
    embedding = np.load(path)
    assert embedding.shape[0] == adata.n_obs, f"{path.name}: {embedding.shape[0]} rows, expected {adata.n_obs}"
    print(f"Scoring {name} ({embedding.shape[1]} dimensions)...")

    metrics, sweep, predictions = evaluate_embedding(embedding, labels)
    metric_rows[name] = metrics
    sweeps.append(sweep.assign(embedding=name))

    summary = per_type_summary(predictions, labels)
    per_type[f"{name}_accuracy"] = summary["accuracy"]
    per_type[f"{name}_most_confused_with"] = summary["most_confused_with"]
    per_type[f"{name}_confused_share"] = summary["confused_share"]
    confusion_table(predictions, labels).to_csv(RESULTS_DIR / f"knn_confusion_{name}.csv", float_format="%.3f")

    # Which types the clustering merges or splits: at the best-ARI resolution, and at the first
    # resolution that gives at least one cluster per cell type.
    matched_resolution = sweep.loc[sweep["n_clusters"] >= len(set(labels)), "resolution"].min()
    for resolution in [metrics["ari_resolution"], matched_resolution]:
        composition = cluster_composition(adata, leiden_clusters(embedding, resolution))
        composition.to_csv(RESULTS_DIR / f"leiden_composition_{name}_res{resolution}.csv", float_format="%.2f")

    umaps[title] = umap_coordinates(embedding)

metrics_table = pd.DataFrame.from_dict(metric_rows, orient="index")
metrics_table.index.name = "embedding"
metrics_table.to_csv(RESULTS_DIR / "metrics.csv", float_format="%.4f")
pd.concat(sweeps).to_csv(RESULTS_DIR / "leiden_sweep.csv", index=False, float_format="%.4f")
per_type.index.name = "cell_type"
per_type.to_csv(RESULTS_DIR / "knn_per_cell_type.csv", float_format="%.3f")
# Supports the biological interpretation in the README: which markers each cell type expresses.
marker_detection(adata).to_csv(RESULTS_DIR / "marker_detection.csv", float_format="%.2f")

plot_umaps(umaps, labels, FIGURES_DIR / "umap_pca_vs_geneformer.png")
plot_metrics(metrics_table, {name: title for name, (_, title) in EMBEDDINGS.items()}, FIGURES_DIR / "metrics.png")

print("\nMetrics (results/metrics.csv):")
print(metrics_table.round(3).to_string())
print("\nPer-cell-type kNN accuracy (results/knn_per_cell_type.csv):")
print(per_type.round(2).to_string())
