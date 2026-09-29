"""Illustrative, untrained query, key, and value projections."""

import numpy as np


class QKVProjector:
    """Apply three independent linear projections, without biases.

    Weights are initialized once and reused for every token and input.
    A fixed seed makes this demonstration reproducible; it does not replace
    the training needed for meaningful attention weights.
    """

    def __init__(self, dimensions: int = 3, seed: int = 42) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dimensions)
        self.W_Q, self.W_K, self.W_V = rng.normal(
            scale=scale, size=(3, dimensions, dimensions)
        ).astype(np.float32)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Q, K, V, each with shape (number_of_tokens, dimensions)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dimensions:
            raise ValueError(f"X must have shape (number_of_tokens, {self.dimensions})")
        return X @ self.W_Q, X @ self.W_K, X @ self.W_V
