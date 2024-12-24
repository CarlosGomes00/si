import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):

    def __init__(self, l1_penalty: float = 0.1, max_iter: int = 1000, scale: bool = True, patience : int = 5, **kwargs):
        """
        Initialize the Lasso Regression model

        Parameters:
        ----------
        l1_penalty : float
            L1 regularization parameter
        max_iter : int
            Maximum number of iterations
        scale : bool
            Whether or not to scale the data
        patience : int
            The patience parameter for early stopping
        """

        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.max_iter = max_iter
        self.patience = patience

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = []

    def _fit(self, dataset: Dataset) -> 'LassoRegression':

        """
        Estimates the theta and theta_zero coefficients, mean and std

         Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LassoRegression
            The fitted model
        """

        if self.scale:
            self.mean = dataset.X.mean(axis=0)
            self.std = dataset.X.std(axis=0)
            X_scaled = (dataset.X - self.mean) / self.std
        else:
            X_scaled = dataset.X


        m, n = dataset.shape()
        self.theta = np.zeros(n)
        self.theta_zero = 0.0

        i = 0
        early_stopping = 0

        while i < self.max_iter and early_stopping < self.patience:

            y_pred = np.dot(X_scaled, self.theta) + self.theta_zero

            for j in range(n):
                residual = np.dot(X_scaled[:, j], dataset.y - (y_pred - self.theta[j] * X_scaled[:, j]))

                if residual < -self.l1_penalty:
                    self.theta[j] = (residual + self.l1_penalty) / np.sum(X_scaled[:, j] ** 2)
                elif residual > self.l1_penalty:
                    self.theta[j] = (residual - self.l1_penalty) / np.sum(X_scaled[:, j] ** 2)
                else:
                    self.theta[j] = 0.0

            self.theta_zero = np.mean(dataset.y - np.dot(X_scaled, self.theta))

            y_pred = np.dot(X_scaled, self.theta) + self.theta_zero
            cost = np.mean((dataset.y - y_pred) ** 2) + self.l1_penalty * np.sum(np.abs(self.theta))
            self.cost_history.append(cost)

            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the target values based on the fitted model

        Parameters
        ----------
        dataset: Dataset
            The dataset containing features to predict the target variable

        Returns
        -------
        y_pred: np.array
            The predicted values for the target variable
        """

        if self.scale:
            if self.mean is None or self.std is None:
                raise ValueError("Model not fitted yet or scaling parameters missing")
            X_scaled = (dataset.X - self.mean) / self.std
        else:
            X_scaled = dataset.X

        predictions = np.dot(X_scaled, self.theta) + self.theta_zero

        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error (MSE) of the model

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the MSE score
        y_pred: np.ndarray
            The predicted values for the target variable
        Returns
        -------
        mse: float
            The Mean Squared Error (MSE) score
        """

        return mse(dataset.y, predictions)

#TODO Perceber porque não está a passar no ultimo teste