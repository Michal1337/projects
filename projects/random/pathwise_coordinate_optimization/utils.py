from typing import Tuple

import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0)
    X = X / X.shape[0] ** (1 / 2)
    return X


def generate_data(n: int, p: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    corr_matrix = np.full((p, p), rho)
    np.fill_diagonal(corr_matrix, 1)

    predictors = np.random.multivariate_normal(
        mean=np.zeros(p), cov=corr_matrix, size=n
    )

    beta = np.array([(-1) ** j * np.exp(-2 * (j - 1) / 20) for j in range(1, p + 1)])

    Z = np.random.normal(0, 1, size=n)
    k = np.sqrt(np.sum(beta**2)) / 3.0  # Calculate k to achieve SNR = 3.0
    outcomes = np.dot(predictors, beta) + k * Z

    return predictors, outcomes
