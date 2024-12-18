import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from typing import Callable


class KNNRegressor(Model):

    def __init__self(self, k: int, distance: Callable, **kwargs):

        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """

        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        ----------
        self: KNNClassifier
            The fitted model
        """

        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.array:
        """
        Estimates the values based on the k nearest neighbours

        Parameters
        ----------
        dataset: Dataset
            The dataset

        Returns
        ----------
        predictions: np.array
            Array with the predictions
        """

        predictions = np.zeros(dataset.X.shape[0])

        for i in range(dataset.X.shape[0]):
            distances = np.array([self.distance(dataset.X[i], x_train) for x_train in self.dataset.X])

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_values = self.dataset.y[k_indices]

            predictions[i] = np.mean(k_nearest_values)

        return predictions

    def _score(self, dataset: Dataset) -> 'rmse':

        """
        It calculates the error between the estimated and actual values

        Parameters
        ----------
        dataset: Dataset
            The dataset

        Returns
        ----------
        rmse: float
            Value corresponding to the error between y_true and y_pred
        """

        return rmse(dataset.y, self._predict(dataset))
