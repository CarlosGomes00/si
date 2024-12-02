import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1.0, 2.0],
                      [np.nan, 4.0],
                      [5.0, 6.0],
                      [7.0, np.nan]])
        y = np.array([10, 20, 30, 40])

        expected_X = np.array([[1.0, 2.0],
                               [5.0, 6.0]])
        expected_y = np.array([10, 30])

        dataset = Dataset(X, y)

        dataset.dropna()

        np.testing.assert_array_equal(dataset.X, expected_X, "X array did not match expected values")
        np.testing.assert_array_equal(dataset.y, expected_y, "y array did not match expected values")

    def test_fillna(self):
        X = np.array([[1.0, 2.0, np.nan],
                      [np.nan, 4.0, 6.0],
                      [7.0, 8.0, 9.0]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.fillna(0)
        expected_X_0 = np.array([[1.0, 2.0, 0.0],
                                 [0.0, 4.0, 6.0],
                                 [7.0, 8.0, 9.0]])

        np.testing.assert_array_equal(dataset.X, expected_X_0)

        dataset.X = X
        dataset.fillna("mean")
        expected_X_mean = np.array([[1.0, 2.0, 7.5],
                                    [4.0, 4.0, 6.0],
                                    [7.0, 8.0, 9.0]])

        np.testing.assert_array_equal(dataset.X, expected_X_mean)

        dataset.X = X
        dataset.fillna("median")
        expected_X_median = np.array([[1.0, 2.0, 7.5],
                                      [4.0, 4.0, 6.0],
                                      [7.0, 8.0, 9.0]])

        np.testing.assert_array_equal(dataset.X, expected_X_median)

    def test_remove_by_index(self):
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])

        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.remove_by_index(1)

        expected_X = np.array([[1.0, 2.0, 3.0],
                               [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(dataset.X, expected_X)

        expected_y = np.array([1, 3])
        np.testing.assert_array_equal(dataset.y, expected_y)
