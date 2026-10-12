# Notes from using the Helical SDK

What I learned running Geneformer through `helical==3.1.3` for this comparison. Each point was checked against the installed source or by running it.

## Model and settings

- `GeneformerConfig(model_name="gf-12L-38M-i4096", batch_size=1, device="cpu", emb_mode="cell")`, with `emb_layer=-1` (Helical's default, the final layer) and, as a third row, `emb_layer=11` (see the last section).
- 12 transformer layers, hidden size 512, 8 attention heads, 38.0M parameters. Input is `<cls>`, then up to 4,094 genes ranked by expression, then `<eos>`. `emb_mode="cell"` averages the gene-token outputs, leaving out `<cls>`, `<eos>` and padding.
- Vocabulary overlap: 11,842 of the 13,656 PBMC3k genes detected in at least 3 cells (86.7%) have an Ensembl ID in Geneformer's 20,275-token vocabulary. No cell has more than 2,413 detected genes, so none was truncated.
- Runtime: 41 minutes for the 2,638 cells (0.93 s per cell), plus a 2-minute smoke test on 100 cells, on a 4-core Intel i5-1030NG7 laptop, CPU only, inside Docker with 4 PyTorch threads.

## `process_data()` writes onto the AnnData you pass in

Before tokenising, `process_data()` runs a validation step that stores each cell's total count in `obs["total_counts"]` on the object it receives. With the default `gene_names="index"` it also copies the index into `var["index"]`. (With `gene_names="ensembl_id"`, as here, only `obs["total_counts"]` is added.) This is intentional: the totals are how the step checks that `X` holds raw counts, and Helical's own tests assert the column exists afterwards (`ci/tests/test_scgpt/test_scgpt_model.py`). This repo therefore passes `adata.copy()` into Helical.

Two related details:

- The raw-count check looks at per-cell totals, not individual values: a matrix of 0.5s whose rows sum to whole numbers is accepted. `data.py` therefore asserts integer values itself.
- The Geneformer tokenizer divides each cell by its total count (`obs["n_counts"]` if present, otherwise recomputed from `X`) and by each gene's median from the pretraining corpus, then ranks genes within the cell. The per-cell total therefore does not change a cell's tokens; only the per-gene medians do.

## Friction hit, and the fix

1. **No install on an Intel Mac.** `helical==3.1.3` pins `torch==2.10.0`, and PyTorch has not published Intel-Mac wheels since 2.2.2. The `bitsandbytes>=0.48.2` requirement has no Intel-Mac build either. Fix: run in a Linux container (`Dockerfile`). On Linux the PyPI torch wheel pulls in 18 GPU packages (CUDA libraries and Triton), so `pyproject.toml` takes torch from PyTorch's CPU-only index; `2.10.0+cpu` satisfies helical's `==2.10.0` pin.
2. **`KeyError: 'ensembl_id'` from `process_data()` on PBMC3k.** By default the tokenizer maps every Ensembl ID through Geneformer's ID-mapping table and sums genes that end up with the same ID (`collapse_gene_ids=True`). PBMC3k's 2016 annotation has 43 such pairs; for example, TARDBP and the retired ID for RP4-635E18.8 both map to ENSG00000120948. On that path the rebuilt AnnData keeps only `var["ensembl_id_collapsed"]`, but the next line reads `var["ensembl_id"]`. Fix: `geneformer.tk.collapse_gene_ids = False`, a tokenizer option that `GeneformerConfig` does not expose. The IDs here are already unique, so each gene keeps its own ID and only IDs in the vocabulary are used (11,842 genes). The cost: 80 retired IDs are skipped instead of renamed; 37 of them would have added a gene, and 43 would have been summed into a gene already present.
3. **Batch size on CPU.** Each batch is padded to its longest cell. Measured on the first 32 cells with 4 threads:

   | batch_size | seconds per cell | peak memory |
   |---|---|---|
   | 1 | 1.28 | 0.85 GB |
   | 2 | 1.48 | 1.08 GB |
   | 8 | 3.29 | 2.28 GB |
   | 16 | 6.32 | 3.32 GB |

   Embeddings matched across batch sizes to within 2e-6, so this repo uses `batch_size=1` rather than 16. Helical loads Geneformer as `BertForMaskedLM`, so each forward pass also computes 20,275 gene logits per token that embeddings never use; removing that head saved a further 15% at batch size 1, but the model is left as shipped.
4. **Threads in Docker.** Docker presents this 4-core CPU as 8 CPUs (hyper-threads), so PyTorch starts 8 threads. With `OMP_NUM_THREADS=4` (one thread per physical core), embedding at batch size 4 went from 3.0 to 1.6 seconds per cell.

## `emb_layer` counts differently from the original Geneformer

Helical uses `emb_layer` as a Python index into the model's 13 hidden states (the token-embedding layer, then 12 transformer layers):

| `emb_layer` | Helical 3.1.3 | original Geneformer `EmbExtractor` |
|---|---|---|
| `-1` | final layer (default) | second-to-last layer (default) |
| `0` | token-embedding layer, before any transformer block | final layer |
| `11` | second-to-last layer | not allowed |
| `-2`, `12` | rejected with `ValueError` | not allowed |

The original documentation recommends the second-to-last layer: "The last layer is most specifically weighted to optimize the given learning objective. Generally, it is best to extract the 2nd to last layer to get a more general representation." To get that layer in Helical this repo uses `emb_layer=11`, since `-2` is rejected.
