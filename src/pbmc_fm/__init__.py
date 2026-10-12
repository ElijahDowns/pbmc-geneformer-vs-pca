"""Compare Geneformer (via the Helical SDK) and Scanpy PCA embeddings on PBMC3k."""

import random

import numpy as np
import torch

SEED = 0


def set_seeds(seed=SEED):
    """Fix the Python, NumPy and PyTorch random seeds so every run gives the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
