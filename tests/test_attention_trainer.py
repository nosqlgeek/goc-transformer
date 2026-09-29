import unittest

import numpy as np

from transformer.attention_trainer import AttentionTrainer
from transformer.qkv_projector import QKVProjector


class AttentionTrainerTests(unittest.TestCase):
    def setUp(self):
        self.trainer = AttentionTrainer(QKVProjector(), vocabulary_size=4)
        self.X = np.array([[0.2, 0.6, -0.1], [0.8, -0.3, 0.4], [-0.2, 0.1, 0.9]])
        self.targets = np.array([1, 2, 3])

    def test_backprop_matches_finite_differences(self):
        _, gradients = self.trainer.loss_and_gradients(self.X, self.targets)
        epsilon = 1e-5
        for name, parameter in self.trainer.parameters.items():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus, _ = self.trainer.loss_and_gradients(self.X, self.targets)
                parameter[index] = original - epsilon
                minus, _ = self.trainer.loss_and_gradients(self.X, self.targets)
                parameter[index] = original
                self.assertAlmostEqual(
                    gradients[name][index], (plus - minus) / (2 * epsilon), places=7,
                    msg=f"{name}{index}",
                )

    def test_future_tokens_cannot_change_earlier_predictions(self):
        before = self.trainer.predict(self.X)
        changed = self.X.copy()
        changed[-1] = [10, -20, 30]
        np.testing.assert_allclose(self.trainer.predict(changed)[:-1], before[:-1])
        np.testing.assert_allclose(before.sum(axis=1), 1)

    def test_training_reduces_loss_and_updates_all_projections(self):
        original = {name: value.copy() for name, value in self.trainer.parameters.items()}
        history = self.trainer.fit([(self.X, self.targets)], epochs=500)
        self.assertLess(history[-1], history[0] * 0.7)
        for name in ("W_Q", "W_K", "W_V"):
            self.assertGreater(np.linalg.norm(self.trainer.parameters[name] - original[name]), 0.01)

    def test_invalid_training_inputs(self):
        for examples in ([], [(np.empty((0, 3)), np.array([], dtype=int))]):
            with self.assertRaises(ValueError):
                self.trainer.fit(examples)
        with self.assertRaises(ValueError):
            self.trainer.loss_and_gradients(self.X, np.array([0, 1, 9]))


if __name__ == "__main__":
    unittest.main()
