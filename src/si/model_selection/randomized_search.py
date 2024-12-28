import numpy as np
from typing import Dict, Callable, Union, List
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def random_search_cv(model, dataset: Dataset, param_grid: Dict[str, List], scoring: Callable = None, cv: int = 3, n_iter: int = 10) -> Dict[str, Union[float, Dict]]:

    """
    Randomized hyperparameter search with cross-validation

    Parameters:
    ----------
    model : object
        The model to be fitted
    dataset : Dataset
        The dataset for validation
    param_grid : dict
        Dictionary with the names of the hyperparameters and their possible values
    scoring : callable, optional
        Function to compute the score of the model
    cv : int
        Number of folds for cross-validation
    n_iter : int
        Number of random combinations of hyperparameters to test

    Returns:
    --------
    Dict[str, Union[float, Dict]]:
        A dictionary with the best hyperparameters, scores, and details
    """

    results = {
        "hyperparameters": [],
        "scores": [],
        "best_hyperparameters": None,
        "best_score": -np.inf
    }

    for _ in range(n_iter):
        params = {key: np.random.choice(values) for key, values in param_grid.items()}

        for param, value in params.items():
            setattr(model, param, value)

        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(scores)

        results["hyperparameters"].append(params)
        results["scores"].append(mean_score)

        if mean_score > results["best_score"]:
            results["best_score"] = mean_score
            results["best_hyperparameters"] = params

    return results