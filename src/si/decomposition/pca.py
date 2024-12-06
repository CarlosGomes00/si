import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer

class PCA(Transformer):

    def __init__(self, n_components : int, **kwargs):
        """
        Parameters
        ----------
        n_components : int
            number of components

        Estimated Parameters
        ----------
        mean = int
            mean of the samples
        components = np.array
            the principal components (a matrix where each row is an eigenvector corresponding to a principal component)
        explained_variance = np.array
            the amount of variance explained
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None


    def _fit(self, dataset: Dataset) -> 'PCA':

        """
        Estimates the mean, principal components and explained variance of the data set

        Parameters
        ----------
        dataset: Dataset
            Dataset object containing the data to be reduced

        Returns
        -------
        self: PCA
            Returns the adjusted PCA instance
        """

        if self.n_components > dataset.X.shape[1]:
            raise ValueError(
                f"n_components ({self.n_components}) cant be greater than the number of features ({dataset.X.shape[1]})."
            )

        self.mean = np.mean(dataset.X, axis=0)
        centered_data = dataset.X - self.mean

        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

        total_variance = np.sum(eigenvalues)

        self.explained_variance = eigenvalues[:self.n_components] / total_variance

        return self

    def _transform(self, dataset: Dataset) -> np.ndarray:

        """
        Transforms data into principal component space

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the data to be reduced

        Returns
        -------
        x_reduced : np.ndarray
            Data projected into the reduced space of principal components
        """
        if self.mean is None or self.components is None:
            raise ValueError("The PCA has not been adjusted. Call _fit before _transform")

        centered_data = dataset.X - self.mean

        x_reduced = np.dot(centered_data, self.components)

        return x_reduced

#TODO Perceber o que está errado e ver Notebook