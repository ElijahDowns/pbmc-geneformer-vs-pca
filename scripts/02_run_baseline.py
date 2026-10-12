"""Step 2: compute the 50-dimensional Scanpy PCA embedding of the labelled cells."""

from pathlib import Path

import anndata as ad
import numpy as np

from pbmc_fm import set_seeds
from pbmc_fm.baseline import N_COMPONENTS, pca_embedding

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "pbmc3k_labelled.h5ad"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "pca_embedding.npy"

set_seeds()
RESULTS_DIR.mkdir(exist_ok=True)

adata = ad.read_h5ad(INPUT_PATH)
embedding = pca_embedding(adata)
assert embedding.shape == (adata.n_obs, N_COMPONENTS)

np.save(OUTPUT_PATH, embedding.astype(np.float32))
print(f"PCA embedding: {embedding.shape[0]} cells x {embedding.shape[1]} components")
print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
