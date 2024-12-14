from unittest import TestCase
from si.io.data_file import read_data_file
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import random_search_cv
import numpy as np

class TestRandomSearchCV(TestCase):

    def setUp(self):
        self.csv_file = '/Users/carla/PycharmProjects/Mestrado/SI_ML/Sistemas_Inteligentes_ML/si/datasets/breast_bin/breast-bin.csv'
        self.dataset = read_data_file(filename = self.csv_file, label=True, sep=",")

    def test_random_search_cv(self):
        model = LogisticRegression()

        param_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': [1000, 2000, 200]
        }

        results = random_search_cv(
            model=model,
            dataset=self.dataset,
            param_grid=param_grid,
            scoring=None,
            cv=3,
            n_iter=10
        )


        self.assertEqual(len(results["best_params"]), len(param_grid))


        best_params = results["best_params"]
        self.assertTrue("l2_penalty" in best_params)
        self.assertTrue("alpha" in best_params)
        self.assertTrue("max_iter" in best_params)


        best_score = results["best_score"]
        self.assertGreaterEqual(best_score, 0.5)

#TODO Corrigir testes