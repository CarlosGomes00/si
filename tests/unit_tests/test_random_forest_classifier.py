from unittest import TestCase
from si.io.data_file import read_data_file
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import random_search_cv
from datasets import DATASETS_PATH
import numpy as np
import os

class TestRandomSearchCV(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.model = LogisticRegression()
        self.param_grid = {'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': [1000, 2000, 200]}

    def test_random_search_cv(self):

        results = random_search_cv(
            model= self.model,
            dataset= self.dataset,
            param_grid= self.param_grid,
            scoring= None,
            cv=3,
            n_iter=10
        )

        best_params = results["best_hyperparameters"]
        for param in best_params:
            self.assertIn(param, self.param_grid, f"Parameter ‘{param}’ not found in parameter grid")

        self.assertTrue("l2_penalty" in best_params)
        self.assertTrue("alpha" in best_params)
        self.assertTrue("max_iter" in best_params)


        best_score = results["best_score"]
        self.assertGreaterEqual(best_score, 0.5,
                                'The best score must be greater than or equal to 0.5 to be considered valid')

