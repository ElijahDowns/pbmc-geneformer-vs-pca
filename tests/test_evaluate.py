"""Metrics on toy embeddings: perfect for well-separated clusters, near chance for random noise."""

import numpy as np
import pytest

from pbmc_fm.evaluate import evaluate_embedding, per_type_summary

CELL_TYPES = np.array(["A", "B", "C", "D"])
CELLS_PER_TYPE = 50


def test_perfectly_separable_clusters_score_one():
    rng = np.random.default_rng(0)
    labels = np.repeat(CELL_TYPES, CELLS_PER_TYPE)
    # One tight cluster per label, centred 100 units apart along different axes.
    centres = 100 * np.eye(len(CELL_TYPES), 10)
    embedding = np.repeat(centres, CELLS_PER_TYPE, axis=0) + rng.normal(size=(len(labels), 10))

    metrics, _, _ = evaluate_embedding(embedding, labels)

    assert metrics["knn_accuracy"] == 1.0
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["silhouette_cosine"] > 0.9


def test_random_noise_scores_low():
    rng = np.random.default_rng(0)
    labels = rng.choice(CELL_TYPES, size=len(CELL_TYPES) * CELLS_PER_TYPE)
    embedding = rng.normal(size=(len(labels), 10))

    metrics, _, _ = evaluate_embedding(embedding, labels)

    # Guessing gives about 0.25 accuracy with four balanced labels; ARI, NMI, silhouette ~0.
    assert metrics["knn_accuracy"] < 0.4
    assert metrics["ari"] < 0.1
    assert metrics["nmi"] < 0.15
    assert abs(metrics["silhouette_cosine"]) < 0.1


def test_per_type_summary_reports_the_confused_type():
    labels = np.array(["A", "A", "A", "A", "B", "B"])
    predictions = np.array(["A", "A", "A", "B", "B", "B"])

    summary = per_type_summary(predictions, labels)

    assert summary.loc["A", "accuracy"] == 0.75
    assert summary.loc["A", "most_confused_with"] == "B"
    assert summary.loc["B", "accuracy"] == 1.0
    assert summary.loc["B", "most_confused_with"] == ""
