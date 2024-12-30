import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset

class StackingClassifier(Model):


    def __init__(self, models, final_model, **kwargs):
        """
        Initialize the Stacking Classifier model

        Parameters:
        ----------
        models : List of Models
            Initial set of models
        final_model : Model
            The model to make the final prediction
        """
        super().__init__(**kwargs)
        self.final_model = final_model
        self.models = models

    def _fit(self, dataset : Dataset) -> 'StackingClassifier':
        """
        Train the ensemble models and the final model

        Parameters
        ----------
        dataset: Dataset
            The dataset to train the models

        Returns
        -------
        self: StackingClassifier
            The fitted model
        """
        predictions = []
        for model in self.models:
            model._fit(dataset)
            pred = model._predict(dataset)
            predictions.append(pred)

        predictions = np.array(predictions).T
        stacked_dataset = Dataset(X=predictions, y=dataset.y)

        self.final_model._fit(stacked_dataset)

        return self


    def _predict(self, dataset : Dataset) -> np.ndarray:
        """
        Predict the labels using the ensemble models and the final model

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target variable for

        Returns
        -------
        final_predictions: np.ndarray
            The predicted labels for the dataset
        """
        if not getattr(self, 'is_fitted', False):
            raise ValueError("The model must be fitted before predicting")

        predictions = [model._predict(dataset) for model in self.models]
        predictions = np.array(predictions).T
        stacked_dataset = Dataset(X=predictions, y=None)

        final_predictions = self.final_model._predict(stacked_dataset)

        return final_predictions

    def _score(self, dataset: Dataset, final_predictions: np.ndarray) -> float:
        """
        Compute the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the features and true labels
        final_predictions: np.ndarray
            Predicted labels for the dataset

        Returns
        -------
        accuracy: float
            The accuracy of the model, calculated as the proportion of correct predictions
        """
        if not getattr(self, 'has_predicted', False):
            raise ValueError("The model must be used to predict before scoring")

        if len(final_predictions) != len(dataset.y):
            raise ValueError("The number of predictions must match the number of true labels")

        accuracy = np.mean(final_predictions == dataset.y)
        return accuracy