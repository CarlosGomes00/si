import numpy as np


def cosine_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:

    """
    Calculates the Cosine distance between a single sample x and multiple samples y.

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Cosine distance for each point in y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)

    if x.ndim != 1:
        raise ValueError("x should be a 1D array")
    if y.ndim != 2:
        raise ValueError("y should be a 2D array")
    if x.shape[0] != y.shape[1]:
        raise ValueError(f"Dimension mismatch: x has {x.shape[0]} elements, but y expects vectors with {y.shape[1]} elements.")

    if x_norm == 0 or np.any(y_norm == 0):
        raise ValueError("One of the vectors has zero magnitude so cosine distance is undefined.")

    return 1 - (np.dot(x, y.T) / (x_norm * y_norm))