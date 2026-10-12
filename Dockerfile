# CPU-only Linux environment for machines where helical cannot be installed natively
# (e.g. Intel Macs: helical pins torch==2.10.0, which has no Intel-Mac wheel).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

# Install the locked dependencies into /opt/venv, using the image's Python 3.12.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1
WORKDIR /work
COPY pyproject.toml uv.lock ./
# The project itself is not installed: the repo is mounted at /work at run time instead.
RUN uv sync --frozen --no-install-project --no-cache

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/work/src \
    # Keep numba's compiled code (UMAP, QC metrics) in the cache volume so it is built only once.
    NUMBA_CACHE_DIR=/root/.cache/numba
