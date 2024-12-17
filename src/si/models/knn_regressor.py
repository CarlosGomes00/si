import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
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
        -------
        self: KNNClassifier
            The fitted model
        """

        self.dataset = dataset
        return self