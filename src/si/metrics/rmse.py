import numpy as np


def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the difference between the estimated and the actual value of y

    Parameters
    ----------
    y_true : array
        real values for y
    y_pred : array
        predicted values for y

    Returns
    -------
    rmse : float
        value corresponding to the error between y_true and y_pred
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred must have the same shape")

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
