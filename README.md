# Geneformer vs PCA on PBMC3k

**Question:** do zero-shot Geneformer cell embeddings, generated through the Helical SDK, separate known PBMC cell types better than a standard Scanpy PCA embedding?

## Setup

Python 3.12, CPU only. Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`.

### Docker (used for the committed results)

The results here were produced on an Intel Mac, where `helical==3.1.3` cannot be installed natively: it pins `torch==2.10.0`, which has no Intel-Mac wheel (details in [SDK_NOTES.md](SDK_NOTES.md)). The Dockerfile installs the locked dependencies into a Linux Python 3.12 image, using the CPU-only PyTorch build.

```bash
docker build -t pbmc-fm .
docker run --rm -v "$PWD":/work -v pbmc-fm-cache:/root/.cache pbmc-fm \
    python -c "from helical.models.geneformer import Geneformer, GeneformerConfig"
```

The `pbmc-fm-cache` volume keeps Helical's downloaded model weights between runs.

### Native (Apple Silicon on macOS 14+, or Linux)

```bash
uv sync
uv run python -c "from helical.models.geneformer import Geneformer, GeneformerConfig"
```

Not tested on a native machine: `uv.lock` resolves for both platforms, but only the Docker route was run for this repo.
