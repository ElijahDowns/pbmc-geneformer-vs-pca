"""Data preparation: barcode alignment, integer counts, eight labels.

The PBMC3k tests download the dataset (~30 MB) into data/ on first run; no model is downloaded.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pbmc_fm.data import (
    EXPECTED_N_CELLS,
    attach_labels,
    is_integer_counts,
    load_labels,
    prepare_labelled_pbmc3k,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def toy_raw_counts():
    """Four cells x three genes; each cell's counts are distinct so rows can be traced."""
    counts = sp.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))
    return ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4"]),
        var=pd.DataFrame(index=["gene1", "gene2", "gene3"]),
    )


@pytest.fixture(scope="module")
def pbmc3k():
    return prepare_labelled_pbmc3k(DATA_DIR)


def test_attach_labels_keeps_each_barcode_with_its_counts():
    raw = toy_raw_counts()
    labels = pd.Series(["T cell", "B cell"], index=["cell3", "cell1"])

    labelled = attach_labels(raw, labels)

    assert list(labelled.obs_names) == ["cell3", "cell1"]
    assert list(labelled.obs["cell_type"]) == ["T cell", "B cell"]
    assert np.array_equal(labelled.X.toarray(), raw.X.toarray()[[2, 0]])
    # The input object must not be modified.
    assert "cell_type" not in raw.obs


def test_attach_labels_rejects_a_barcode_missing_from_the_raw_data():
    labels = pd.Series(["T cell"], index=["not-a-barcode"])
    with pytest.raises(AssertionError):
        attach_labels(toy_raw_counts(), labels)


def test_is_integer_counts():
    assert is_integer_counts(sp.csr_matrix([[0, 3], [1, 0]], dtype=np.float32))
    assert not is_integer_counts(np.array([[0.5, 1.0]]))
    assert not is_integer_counts(np.array([[-1.0, 2.0]]))


def test_pbmc3k_cells_and_labels_line_up(pbmc3k):
    labels = load_labels(DATA_DIR)
    assert pbmc3k.n_obs == len(labels) == EXPECTED_N_CELLS
    assert list(pbmc3k.obs_names) == list(labels.index)
    assert list(pbmc3k.obs["cell_type"]) == list(labels)


def test_pbmc3k_counts_are_integers(pbmc3k):
    assert is_integer_counts(pbmc3k.X)


def test_pbmc3k_has_eight_cell_types(pbmc3k):
    assert pbmc3k.obs["cell_type"].nunique() == 8
