from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

import numpy as np
import os

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_split_output_type(self):

        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertIsInstance(train, type(self.dataset), "Train dataset isn't a Dataset type")
        self.assertIsInstance(test, type(self.dataset), "Test dataset isn't a Dataset type")

    def test_stratified_split_proportion(self):

        test_size = 0.2
        train, test = stratified_train_test_split(self.dataset, test_size=test_size, random_state=123)

        total_samples = len(self.dataset.y)
        expected_test_size = int(total_samples * test_size)

        self.assertEqual(len(test.y), expected_test_size, "The size of the test set isn't correct")
        self.assertEqual(len(train.y), total_samples - expected_test_size,
                         "The size of the train set isn't correct")

    def test_stratified_split_random_state(self):

        train1, test1 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        train2, test2 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        np.testing.assert_array_equal(train1.X, train2.X, "Training data isn't consistent")
        np.testing.assert_array_equal(test1.X, test2.X, "Test data isn't consistent")

    def test_stratified_split_stratification(self):

        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        unique_labels, original_counts = np.unique(self.dataset.y, return_counts=True)
        original_ratios = original_counts / len(self.dataset.y)

        _, train_counts = np.unique(train.y, return_counts=True)
        _, test_counts = np.unique(test.y, return_counts=True)

        train_ratios = train_counts / len(train.y)
        test_ratios = test_counts / len(test.y)

        np.testing.assert_allclose(original_ratios, train_ratios, atol=0.01,
                                   err_msg="The proportions in the training set don't match the originals")
        np.testing.assert_allclose(original_ratios, test_ratios, atol=0.01,
                                   err_msg="The proportions in the test set don't match the originals")


