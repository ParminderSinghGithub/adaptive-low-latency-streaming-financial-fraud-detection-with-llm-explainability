"""
Reproducibility and random seed management utility.
Enforces deterministic random seed initialization across standard Python and NumPy random generators.
"""

import random
from typing import Dict
import numpy as np


def set_seed(seed: int) -> Dict[str, bool]:
    """
    Set global random seeds for deterministic execution.

    Args:
        seed: Non-negative integer random seed.

    Returns:
        Dict[str, bool] indicating which library seeds were configured.

    Raises:
        ValueError: If seed is negative or not an integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got: {seed}")

    status = {
        "random": False,
        "numpy": False,
    }

    # Seed Python standard random module
    random.seed(seed)
    status["random"] = True

    # Seed NumPy random generator
    np.random.seed(seed)
    status["numpy"] = True

    return status
