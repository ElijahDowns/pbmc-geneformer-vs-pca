"""Zero-shot Geneformer cell embeddings through the Helical SDK, on CPU."""

from helical.models.geneformer import Geneformer, GeneformerConfig

MODEL_NAME = "gf-12L-38M-i4096"
# One cell per forward pass. Each batch is padded to its longest cell, so on a 4-core CPU batch 16
# took 6.3 s/cell and 3.3 GB versus 1.3 s/cell and 0.85 GB for batch 1, with embeddings equal to
# within 2e-6 (benchmark in SDK_NOTES.md).
BATCH_SIZE = 1


def load_geneformer(emb_layer=-1, model_name=MODEL_NAME):
    """Load pretrained Geneformer for mean-pooled cell embeddings (weights download on first use).

    Helical uses `emb_layer` as an index into the model's hidden states, so -1 is the final layer.
    """
    config = GeneformerConfig(
        model_name=model_name,
        batch_size=BATCH_SIZE,
        device="cpu",
        emb_mode="cell",
        emb_layer=emb_layer,
        # Copy obs["barcode"] into the tokenised dataset so the cell order can be checked afterwards.
        custom_attr_name_dict={"barcode": "barcode"},
    )
    geneformer = Geneformer(configurer=config)
    # PBMC3k's Ensembl IDs are unique, but 43 pairs of them map to the same current ID in
    # Geneformer's mapping table. Helical's default ID-collapsing step then fails with
    # KeyError: 'ensembl_id' (see SDK_NOTES.md). Switching it off keeps each gene's own ID;
    # genes whose ID is not in the vocabulary are skipped, as they would be anyway.
    geneformer.tk.collapse_gene_ids = False
    return geneformer


def vocabulary_overlap(adata, geneformer):
    """Number of the dataset's genes whose Ensembl ID is in Geneformer's token vocabulary."""
    vocabulary = list(geneformer.tk.gene_token_dict.keys())
    return int(adata.var["gene_ids"].isin(vocabulary).sum())


def embed_cells(adata, geneformer):
    """Return a (n_cells, embedding_dim) array of Geneformer cell embeddings, rows in `adata` order."""
    # Work on a copy so the caller's AnnData does not gain the columns added below.
    adata = adata.copy()
    # Geneformer's vocabulary is keyed by Ensembl ID. Passing the IDs directly means Helical does
    # no symbol-to-ID mapping, so no network call to Ensembl and no mapping losses.
    adata.var["ensembl_id"] = adata.var["gene_ids"]
    adata.obs["barcode"] = adata.obs_names.to_numpy()

    # process_data writes obs["total_counts"] onto the AnnData it receives, so it gets its own copy.
    dataset = geneformer.process_data(adata.copy(), gene_names="ensembl_id")
    assert list(dataset["barcode"]) == list(adata.obs_names), "tokenised cells are not in input order"

    embeddings = geneformer.get_embeddings(dataset)
    assert embeddings.shape == (adata.n_obs, geneformer.config["embsize"])
    return embeddings
