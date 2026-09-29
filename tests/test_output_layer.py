import unittest

import numpy as np

from transformer.output_layer import OutputLayer
from transformer.attention_trainer import AttentionTrainer
from transformer.qkv_projector import QKVProjector
from transformer.self_attention import SelfAttention


class OutputLayerTests(unittest.TestCase):
    def test_known_scores_probabilities_and_token_mapping(self):
        layer = OutputLayer(np.array([[1., 0., -1.], [0., 2., 1.]]), np.array([0., 1., 0.]))
        Y = np.array([[1., 2.]])
        np.testing.assert_array_equal(layer.logits(Y), [[1., 5., 1.]])
        probabilities = layer.predict(Y)
        np.testing.assert_allclose(probabilities, [[1/(2+np.exp(4)), np.exp(4)/(2+np.exp(4)), 1/(2+np.exp(4))]])
        self.assertEqual(["cat", "milk", "dog"][probabilities[0].argmax()], "milk")

    def test_large_scores_empty_input_and_normalization(self):
        layer = OutputLayer(np.array([[1000., 1001.]]), np.zeros(2))
        np.testing.assert_allclose(layer.predict([[1.]]).sum(axis=1), 1)
        self.assertTrue(np.isfinite(layer.predict([[1.]])).all())
        self.assertEqual(layer.predict(np.empty((0, 1))).shape, (0, 2))

    def test_demo_output_matches_trainer_after_learning(self):
        projector = QKVProjector()
        trainer = AttentionTrainer(projector, 3)
        X = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        trainer.fit([(X, np.array([1, 2]))], epochs=5)
        Q, K, V = projector.project(X)
        Y = SelfAttention().calculate(Q, K) @ V
        np.testing.assert_allclose(trainer.output_layer.predict(Y), trainer.predict(X))


if __name__ == "__main__":
    unittest.main()
