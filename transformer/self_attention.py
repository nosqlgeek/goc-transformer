"""Scaled dot-product attention weights for one causal attention head."""

import numpy as np


class SelfAttention:
    """Calculate how strongly each token attends to itself and earlier tokens."""

    def calculate(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return A with shape (tokens, tokens); each non-empty row sums to 1."""
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        if Q.ndim != 2 or K.shape != Q.shape or Q.shape[1] == 0:
            raise ValueError("Q and K must have matching (tokens, dimensions) shapes")
        if not np.isfinite(Q).all() or not np.isfinite(K).all():
            raise ValueError("Q and K must contain finite values")
        if len(Q) == 0:
            return np.empty((0, 0), dtype=np.float64)

        scores = Q @ K.T / np.sqrt(Q.shape[1])
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores[future] = -np.inf
        # Subtract the row maximum to keep softmax numerically stable.
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)
