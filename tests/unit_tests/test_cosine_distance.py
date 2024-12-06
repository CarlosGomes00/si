import unittest
import numpy as np
from si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(unittest.TestCase):

    def test_samesize_vectors(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [1, 2, 3]])
        result = cosine_distance(x, y)
        expected = np.array([0.0, 0.0])
        np.testing.assert_almost_equal(result, expected)

    def test_orthogonal_vectors(self):
        x = np.array([1, 0])
        y = np.array([[0, 1], [0, 1]])
        result = cosine_distance(x, y)
        expected = np.array([1.0, 1.0])
        np.testing.assert_almost_equal(result, expected)

    def test_partial_similarity(self):
        x = np.array([1, 1])
        y = np.array([[1, 0], [0.5, 0.2]])
        result = cosine_distance(x, y)
        expected = np.array([0.29, 0.08])
        np.testing.assert_almost_equal(result, expected, decimal = 2)

    def test_proportional_vectors(self):
        x = np.array([1, 1])
        y = np.array([[2, 2], [2, 2]])
        result = cosine_distance(x, y)
        expected = np.array([0.0, 0.0])
        np.testing.assert_almost_equal(result, expected)

    def test_zero_vector(self):
        x = np.array([0, 0, 0])
        y = np.array([[1, 2, 3], [2, 3, 4]])
        with self.assertRaises(ValueError):
            cosine_distance(x, y)


if __name__ == '__main__':
    unittest.main()
