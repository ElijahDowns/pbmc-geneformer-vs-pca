"""Step 1: build the labelled PBMC3k AnnData (raw counts + cell types) that every later step reads."""

from pathlib import Path

from pbmc_fm.data import prepare_labelled_pbmc3k

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "pbmc3k_labelled.h5ad"

DATA_DIR.mkdir(exist_ok=True)
adata = prepare_labelled_pbmc3k(DATA_DIR)

print(f"Cells: {adata.n_obs}   Genes detected in >= 3 cells: {adata.n_vars}")
print("\nCells per type:")
print(adata.obs["cell_type"].value_counts().to_string())
print("\nPer-cell QC (median):")
print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].median().round(2).to_string())

adata.write_h5ad(OUTPUT_PATH)
print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)}")
