"""Map attention outputs to next-token scores and probabilities."""

import numpy as np


class OutputLayer:
    """Use output weights whose columns follow the vocabulary's token-ID order."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.weights = np.asarray(weights)
        self.bias = np.asarray(bias)
        if (
            self.weights.ndim != 2
            or min(self.weights.shape) < 1
            or self.bias.shape != (self.weights.shape[1],)
        ):
            raise ValueError("Expected weights (dimensions, vocabulary_size) and bias (vocabulary_size,)")

    def logits(self, Y: np.ndarray) -> np.ndarray:
        """Return a score per vocabulary token for each input row."""
        Y = np.asarray(Y)
        if Y.ndim != 2 or Y.shape[1] != self.weights.shape[0]:
            raise ValueError("Y must have shape (tokens, output_layer_dimensions)")
        return Y @ self.weights + self.bias

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """Return row-wise next-token probabilities using stable softmax."""
        scores = self.logits(Y)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
