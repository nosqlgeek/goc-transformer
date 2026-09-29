import unittest

import numpy as np

from transformer.self_attention import SelfAttention


class SelfAttentionTests(unittest.TestCase):
    def test_equal_scores_are_uniform_over_available_tokens(self):
        A = SelfAttention().calculate(np.zeros((3, 3)), np.zeros((3, 3)))
        np.testing.assert_allclose(A, [[1, 0, 0], [.5, .5, 0], [1/3, 1/3, 1/3]])

    def test_scaled_scores_and_stability(self):
        Q = np.array([[0, 0], [np.sqrt(2), 0]])
        K = np.array([[1000, 0], [1001, 0]])
        A = SelfAttention().calculate(Q, K)
        np.testing.assert_allclose(A[1], [1 / (1 + np.e), np.e / (1 + np.e)])
        np.testing.assert_allclose(A.sum(axis=1), 1)

    def test_empty_and_single_token(self):
        attention = SelfAttention()
        self.assertEqual(attention.calculate(np.empty((0, 3)), np.empty((0, 3))).shape, (0, 0))
        np.testing.assert_array_equal(attention.calculate([[2, 3]], [[4, 5]]), [[1]])

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            SelfAttention().calculate(np.zeros((2, 3)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
