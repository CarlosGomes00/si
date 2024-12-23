from unittest import TestCase
import numpy as np
import os
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.metrics.rmse import rmse
from si.model_selection.split import train_test_split
from si.statistics.euclidean_distance import euclidean_distance


class TestKNNRegressor(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        self.knn = KNNRegressor(k=2, distance=euclidean_distance)

        self.knn._fit(self.train_dataset)


    def test_fit(self):
        self.knn.fit(self.train_dataset)
        self.assertIsNotNone(self.knn.dataset, "The model was not fitted")

    """
    def test_predict(self):
        self.knn.fit(self.train_dataset)
        predictions = self.knn.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0], 'The number of predictions doesnt match')
    """
    #TODO Perceber porque não funciona e ver mais testes






