"""
Determinism and seed management for reproducible experiments.

This module ensures bit-exact reproducibility across all random number generators
used in the pipeline (random, numpy, torch).
"""

import os
import random
import numpy as np


def set_seeds(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enable deterministic behavior in PyTorch (slower but reproducible)

    Note:
        Deterministic mode may reduce performance but guarantees bit-exact reproducibility.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("Warning: PyTorch not available, skipping torch seed configuration")

    # Python hash seed (for dictionary ordering)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"✓ Random seeds set to {seed} (deterministic={deterministic})")
