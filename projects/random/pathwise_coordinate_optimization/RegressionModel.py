from typing import Dict, Tuple, Union

import numpy as np


class RegressionModel:
    def __init__(self, lam: float, tol: float, max_iter: int) -> None:
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.weights = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = False
    ) -> Tuple[bool, int]:
        self.weights = np.array([1 / X.shape[1]] * X.shape[1])
        for i in range(self.max_iter):
            prev_weights = self.weights.copy()
            if shuffle:
                idx = np.random.permutation(X.shape[1])
            else:
                idx = np.arange(X.shape[1])
            for j in idx:
                beta_hat = self.weights[j] + np.dot(
                    X[:, j], y - np.dot(X, self.weights)
                )

                if beta_hat > 0 and self.lam < np.abs(beta_hat):
                    self.weights[j] = beta_hat - self.lam
                elif beta_hat < 0 and self.lam < np.abs(beta_hat):
                    self.weights[j] = beta_hat + self.lam
                elif self.lam >= np.abs(beta_hat):
                    self.weights[j] = 0

            if np.linalg.norm(self.weights - prev_weights) < self.tol:
                return True, i

        return False, i

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights)

    def get_params(self) -> Dict[str, Union[np.ndarray, float]]:
        return {"betas": self.weights, "lambda": self.lam}
