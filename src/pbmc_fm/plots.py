"""Figures: side-by-side UMAPs coloured by cell type, and a bar chart of the metrics."""

import matplotlib

# Render straight to files: there is no screen inside Docker or CI.
matplotlib.use("Agg")

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from pbmc_fm import SEED
from pbmc_fm.evaluate import neighbour_graph

# One fixed colour per cell type, identical in every panel. Types that sit next to each other
# in PBMC embeddings (CD4 T / CD8 T / NK; the monocytes and dendritic cells) get the pairs that
# stay most distinguishable, including under red-green colour blindness.
CELL_TYPE_COLOURS = {
    "CD4 T cells": "#2a78d6",
    "CD8 T cells": "#1baf7a",
    "NK cells": "#008300",
    "B cells": "#eb6834",
    "CD14+ Monocytes": "#eda100",
    "FCGR3A+ Monocytes": "#e87ba4",
    "Dendritic cells": "#4a3aa7",
    "Megakaryocytes": "#e34948",
}
# Blue, orange, aqua: distinguishable from each other in any combination.
EMBEDDING_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

METRIC_LABELS = {
    "knn_accuracy": "kNN accuracy\n(5-fold CV, k=15)",
    "ari": "ARI\n(best Leiden resolution)",
    "nmi": "NMI\n(best Leiden resolution)",
    "silhouette_cosine": "Silhouette\n(cosine, by cell type)",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def umap_coordinates(embedding):
    """2-D UMAP of an embedding, built from the same 15-neighbour graph as the Leiden clustering."""
    adata = neighbour_graph(embedding)
    sc.tl.umap(adata, random_state=SEED)
    return adata.obsm["X_umap"]


def plot_umaps(umaps, labels, path):
    """One UMAP panel per embedding (title -> coordinates), coloured and labelled by cell type."""
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, len(umaps), figsize=(6.5 * len(umaps), 6.5), facecolor=SURFACE)

    for ax, (title, coordinates) in zip(axes, umaps.items()):
        ax.set_facecolor(SURFACE)
        for cell_type, colour in CELL_TYPE_COLOURS.items():
            in_type = labels == cell_type
            ax.scatter(
                coordinates[in_type, 0],
                coordinates[in_type, 1],
                s=7,
                color=colour,
                linewidths=0,
                label=f"{cell_type} ({in_type.sum()})",
            )

        # Write each type's name at its median position so identity never relies on colour alone.
        for cell_type in CELL_TYPE_COLOURS:
            centre = np.median(coordinates[labels == cell_type], axis=0)
            ax.annotate(
                cell_type,
                centre,
                ha="center",
                va="center",
                fontsize=9,
                color=INK,
                path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)],
            )

        ax.set_title(title, loc="left", fontsize=12, color=INK)
        # UMAP axes have no units, so ticks and frame would only add noise.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        markerscale=3,
        fontsize=10,
        labelcolor=SECONDARY_INK,
    )
    fig.subplots_adjust(bottom=0.12, wspace=0.05)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics, titles, path):
    """Grouped bars: one group per metric, one bar per embedding, with the value above each bar.

    `metrics` is the metrics table (rows = embeddings); `titles` maps each row name to a legend label.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    group_positions = np.arange(len(METRIC_LABELS))
    bar_width = 0.12
    n_embeddings = len(metrics)
    for i, (name, row) in enumerate(metrics.iterrows()):
        # Centre each embedding's bars around the group position, with a small gap between bars.
        offset = (i - (n_embeddings - 1) / 2) * (bar_width + 0.02)
        values = [row[metric] for metric in METRIC_LABELS]
        bars = ax.bar(
            group_positions + offset,
            values,
            width=bar_width,
            color=EMBEDDING_COLOURS[i],
            label=titles[name],
            zorder=2,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5, color=SECONDARY_INK)

    ax.set_xticks(group_positions, list(METRIC_LABELS.values()), fontsize=9.5, color=SECONDARY_INK)
    ax.set_ylim(min(0.0, metrics[list(METRIC_LABELS)].to_numpy().min() - 0.05), 1.08)
    ax.set_ylabel("Score (higher is better)", color=SECONDARY_INK)
    ax.tick_params(axis="both", length=0, colors=SECONDARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=SECONDARY_INK)
    ax.set_title("Same cells, labels and metrics for every embedding", loc="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
