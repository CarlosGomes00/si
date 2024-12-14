from si.base.model import Model
from si.models.decision_tree_classifier import DecisionTreeClassifier
import numpy as np
from si.data.dataset import Dataset


class RandomForestClassifier(Model):

    def __init__(self, n_estimators: int, max_features: int, min_samples_split: int, max_depth: int, mode: str, seed= None, **kwargs):

        """
        Initialize the Random Forest Classifier instance

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_features : int
            Number of features to consider when looking for the best split
        min_samples_split : int
            Minimum number of samples required to split an internal node
        max_depth : int
            Maximum depth of the tree
        mode : str
            Mode of the decision tree ("classification" or "regression")
        seed : int
            Random seed for reproducibility
        """

        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset):

        """
        Fit the RandomForestClassifier model to the dataset

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the training data
        """

        n_samples, n_features = dataset.X.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features > n_features:
            raise ValueError("Number of features cant be greater than the maximum number of features")

        bootstrap_data = None
        feature_indices = None

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size= n_samples, replace= True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            feature_indices = np.random.choice(n_features, size = self.max_features, replace = False)
            bootstrap_X_features = bootstrap_X[:, feature_indices]

            bootstrap_data = Dataset(X = bootstrap_X_features, y = bootstrap_y)

        tree = DecisionTreeClassifier(max_depth= self.max_depth, min_samples_split= self.min_samples_split, mode= self.mode)

        tree._fit(bootstrap_data)

        self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        res_predictions = []

        unique_classes, y_encoded = np.unique(dataset.y, return_inverse=True)
        class_to_int = {label: idx for idx, label in enumerate(unique_classes)}
        int_to_class = {idx: label for idx, label in enumerate(unique_classes)}

        for feature_indices, tree in self.trees:
            subset_dataset = Dataset(X=dataset.X[:, feature_indices], y=dataset.y)
            tree_predictions = tree._predict(subset_dataset)
            predictions_int = np.array([class_to_int[label] for label in tree_predictions])
            res_predictions.append(predictions_int)

        res_predictions = np.array(res_predictions)

        final_predictions = []
        for i in range(res_predictions.shape[1]):
            sample_predictions = res_predictions[:, i]

            most_common = np.bincount(sample_predictions).argmax()
            final_predictions.append(int_to_class[most_common])

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray):
        predictions = self.predict(dataset)
        accuracy = np.mean(predictions == dataset.y)

        return accuracy

#TODO COMPLETAR E CORRIGIR (EX.9)