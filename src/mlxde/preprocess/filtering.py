"""Gene filters deciding which genes carry enough signal to be tested."""

from __future__ import annotations

import numpy as np


class MinimumCountFilter:
    """Keep genes reaching ``min_count`` in at least ``min_samples`` samples."""

    def __init__(self, min_count: int = 10, min_samples: int = 3) -> None:
        if min_count < 0:
            raise ValueError(f"min_count must be non-negative, got {min_count}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")
        self.min_count = min_count
        self.min_samples = min_samples

    def keep(self, counts: np.ndarray) -> np.ndarray:
        if counts.ndim != 2:
            raise ValueError(f"counts must be 2-dimensional, got shape {counts.shape}")
        expressed_samples = np.sum(counts >= self.min_count, axis=1)
        return expressed_samples >= self.min_samples
