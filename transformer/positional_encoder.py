"""Fixed sinusoidal position vectors, requiring no training."""

import numpy as np


class PositionalEncoder:
    """Generate one position vector per token, starting at position zero."""

    def __init__(self, dimensions: int = 3) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def encode(self, length: int) -> np.ndarray:
        """Return a (length, dimensions) matrix, including for empty input."""
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative integer")

        positions = np.arange(length, dtype=np.float64)[:, np.newaxis]
        frequencies = 10000.0 ** (
            -np.arange(0, self.dimensions, 2, dtype=np.float64) / self.dimensions
        )
        angles = positions * frequencies
        encoding = np.empty((length, self.dimensions), dtype=np.float32)
        encoding[:, 0::2] = np.sin(angles)
        # Odd dimensions have one more sine column than cosine columns.
        encoding[:, 1::2] = np.cos(angles[:, : self.dimensions // 2])
        return encoding
