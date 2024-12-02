from si.base.model import Model
from si.models.decision_tree_classifier import DecisionTreeClassifier
import numpy as np
from si.data.dataset import Dataset


class RandomForestClassifier:

    def __init__(self, n_estimators, max_features, min_samples_split, max_depth, mode, seed = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, Dataset, seed) -> 'RandomForestClassifier':

        n_samples, n_features = Dataset.X.shape

        if self.seed is not None:
            self.seed = np.random.seed()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = Dataset[bootstrap_indices]

            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)

        tree = DecisionTreeClassifier(self.max_depth, self.min_sample_split, self.mode)
        tree._fit(bootstrap_data)

        self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset):
        res_predictions = []
        for feature_indices, tree in self.trees:
            prediction = tree._predict(dataset.X[:, feature_indices])
            res_predictions.append(prediction)

        res_predictions = np.array(res_predictions)

        final_predictions = []
        for i in range(res_predictions.shape[1]):
            sample_predictions = res_predictions[:, i]

            most_common = np.bincount(sample_predictions).argmax()
            final_predictions.append(most_common)

        return np.array(final_predictions)

    def _score(self, dataset: Dataset):
        predictions = self.predict(dataset)
        accuracy = np.mean(predictions == dataset.y)

        return accuracy

#TODO COMPLETAR E CORRIGIR (EX.9)