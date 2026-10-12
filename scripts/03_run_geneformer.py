"""Step 3: zero-shot Geneformer cell embeddings through Helical (CPU), after a 100-cell smoke test."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from pbmc_fm import set_seeds
from pbmc_fm.embed_geneformer import (
    BATCH_SIZE,
    MODEL_NAME,
    embed_cells,
    load_geneformer,
    vocabulary_overlap,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "pbmc3k_labelled.h5ad"
RESULTS_DIR = ROOT / "results"
N_SMOKE_TEST_CELLS = 100

set_seeds()
RESULTS_DIR.mkdir(exist_ok=True)
adata = ad.read_h5ad(INPUT_PATH)

# emb_layer=-1: Helical's default, the final transformer layer.
geneformer = load_geneformer(emb_layer=-1)
n_in_vocabulary = vocabulary_overlap(adata, geneformer)
print(
    f"Genes in the Geneformer vocabulary: {n_in_vocabulary} of {adata.n_vars} "
    f"({n_in_vocabulary / adata.n_vars:.1%})"
)

# A problem with the data or the model should surface in about a minute, not after the full run.
start = time.perf_counter()
smoke_embeddings = embed_cells(adata[:N_SMOKE_TEST_CELLS], geneformer)
smoke_seconds = time.perf_counter() - start
estimate_minutes = smoke_seconds / N_SMOKE_TEST_CELLS * adata.n_obs / 60
print(
    f"Smoke test: {smoke_embeddings.shape} in {smoke_seconds:.0f} s "
    f"(full run estimate: {estimate_minutes:.0f} min)"
)

start = time.perf_counter()
embeddings = embed_cells(adata, geneformer)
runtime_seconds = time.perf_counter() - start
print(f"Full run: {embeddings.shape} in {runtime_seconds / 60:.1f} min")

# The same first cells through the same model must give the same embeddings (up to float error
# from different padding in the last smoke-test batch).
assert np.allclose(embeddings[:N_SMOKE_TEST_CELLS], smoke_embeddings, atol=1e-4)

np.save(RESULTS_DIR / "geneformer_embedding.npy", embeddings.astype(np.float32))
run_info = {
    "model_name": MODEL_NAME,
    "emb_layer": -1,
    "emb_mode": "cell",
    "batch_size": BATCH_SIZE,
    "device": "cpu",
    "n_cells": embeddings.shape[0],
    "embedding_dim": embeddings.shape[1],
    "n_genes": adata.n_vars,
    "n_genes_in_vocabulary": n_in_vocabulary,
    "runtime_seconds": round(runtime_seconds, 1),
    "torch_threads": torch.get_num_threads(),
    "helical_version": importlib.metadata.version("helical"),
    "torch_version": torch.__version__,
    "platform": platform.platform(),
}
(RESULTS_DIR / "geneformer_run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
print("Wrote results/geneformer_embedding.npy and results/geneformer_run_info.json")
