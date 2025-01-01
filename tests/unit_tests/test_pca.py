import unittest

from si.decomposition.pca import PCA
from si.io.csv_file import read_csv


class TestPCA(unittest.TestCase):
    def setUp(self):
        csv_file = '/Users/carla/PycharmProjects/Mestrado/SI_ML/Sistemas_Inteligentes_ML/si/datasets/iris/iris.csv'
        self.dataset = read_csv(filename=csv_file, features=True, label=True)

    def test_initialization(self):
        pca = PCA(n_components=2)
        self.assertEqual(pca.n_components, 2)
        self.assertIsNone(pca.mean)
        self.assertIsNone(pca.components)
        self.assertIsNone(pca.explained_variance)

    def test_fit_raises_error(self):
        pca = PCA(n_components=6)
        with self.assertRaises(ValueError):
            pca._fit(self.dataset)

    def test_transform_raises_error(self):
        pca = PCA(n_components=2)
        with self.assertRaises(ValueError):
            pca._transform(self.dataset)







if __name__ == '__main__':
    unittest.main()
