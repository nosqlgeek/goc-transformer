"""A single causal attention head trained by explicit NumPy backpropagation."""

import numpy as np

if __package__:
    from .output_layer import OutputLayer
    from .self_attention import SelfAttention
    from .qkv_projector import QKVProjector
else:
    from output_layer import OutputLayer
    from self_attention import SelfAttention
    from qkv_projector import QKVProjector


DEFAULT_SENTENCES = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat likes milk",
    "the dog likes food",
    "a cat eats food",
    "a dog drinks water",
]


class AttentionTrainer:
    """Learn Q/K/V weights and a vocabulary output layer; keep inputs fixed.

    This deliberately omits multiple heads, feed-forward blocks, normalization,
    and residual connections. It is an attention learning demonstration.
    """

    def __init__(self, projector: QKVProjector, vocabulary_size: int, seed: int = 42):
        if vocabulary_size < 2:
            raise ValueError("Training requires at least two vocabulary words")
        self.projector = projector
        rng = np.random.default_rng(seed)
        self.W_out = rng.normal(
            scale=1 / np.sqrt(projector.dimensions),
            size=(projector.dimensions, vocabulary_size),
        )
        self.bias = np.zeros(vocabulary_size)
        # Float64 makes the explicit gradients easier to check numerically.
        for name in ("W_Q", "W_K", "W_V"):
            setattr(projector, name, getattr(projector, name).astype(np.float64))

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "W_Q": self.projector.W_Q,
            "W_K": self.projector.W_K,
            "W_V": self.projector.W_V,
            "W_out": self.W_out,
            "bias": self.bias,
        }

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.projector.dimensions or len(X) == 0:
            raise ValueError("X must be a non-empty token-by-dimension matrix")
        Q = X @ self.projector.W_Q
        K = X @ self.projector.W_K
        V = X @ self.projector.W_V
        attention = SelfAttention().calculate(Q, K)
        context = attention @ V
        logits = self.output_layer.logits(context)
        return X, Q, K, V, attention, context, logits

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return next-word probabilities at each input position."""
        return self.output_layer.predict(self._forward(X)[-2])

    @property
    def output_layer(self) -> OutputLayer:
        """Expose the current learned output weights without reinitializing them."""
        return OutputLayer(self.W_out, self.bias)

    def loss_and_gradients(self, X: np.ndarray, targets: np.ndarray):
        """Mean next-token cross-entropy and gradients for every parameter."""
        X, Q, K, V, attention, context, logits = self._forward(X)
        targets = np.asarray(targets)
        if (
            targets.shape != (len(X),)
            or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < 0)
            or np.any(targets >= len(self.bias))
        ):
            raise ValueError("targets must contain one valid vocabulary ID per input row")
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -log_probs[np.arange(len(X)), targets].mean()

        # Cross-entropy -> output projection -> weighted values.
        d_logits = np.exp(log_probs)
        d_logits[np.arange(len(X)), targets] -= 1
        d_logits /= len(X)
        d_context = d_logits @ self.W_out.T
        d_attention = d_context @ V.T
        d_V = attention.T @ d_context

        # Differentiate row-wise softmax; masked entries have zero gradients.
        d_scores = attention * (
            d_attention - (d_attention * attention).sum(axis=1, keepdims=True)
        )
        d_scores /= np.sqrt(self.projector.dimensions)
        d_Q = d_scores @ K
        d_K = d_scores.T @ Q
        gradients = {
            "W_Q": X.T @ d_Q,
            "W_K": X.T @ d_K,
            "W_V": X.T @ d_V,
            "W_out": context.T @ d_logits,
            "bias": d_logits.sum(axis=0),
        }
        return float(loss), gradients

    def fit(self, examples, epochs: int = 500, learning_rate: float = 0.05) -> list[float]:
        """Full-batch gradient descent, weighted by number of target tokens.

        Return loss before training and after each update. Modify the supplied
        projector in place, so project(X) subsequently uses the learned weights.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        examples = list(examples)
        if not examples or any(len(X) == 0 for X, _ in examples):
            raise ValueError("Provide non-empty training examples")
        total_tokens = sum(len(X) for X, _ in examples)
        history = []
        for epoch in range(epochs + 1):
            gradients = {name: np.zeros_like(p) for name, p in self.parameters.items()}
            loss = 0.0
            for X, targets in examples:
                example_loss, example_gradients = self.loss_and_gradients(X, targets)
                weight = len(X) / total_tokens
                loss += weight * example_loss
                for name in gradients:
                    gradients[name] += weight * example_gradients[name]
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
                raise ValueError("Training diverged; try a lower learning rate")
            history.append(loss)
            if epoch == epochs:
                break
            # Clip the global gradient norm to keep this small demo stable.
            norm = np.sqrt(sum(np.sum(g * g) for g in gradients.values()))
            scale = min(1.0, 1.0 / max(norm, 1e-12))
            for name, parameter in self.parameters.items():
                parameter -= learning_rate * scale * gradients[name]
        return history
