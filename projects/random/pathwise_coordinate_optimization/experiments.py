import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars, LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from RegressionModel import RegressionModel
from utils import *


def eval(model, X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[float, float]:
    start = time.time()
    model.fit(X, y)
    y_pred = model.predict(X)
    loss = 1 / 2 * ((y - y_pred) ** 2).sum() + lam * np.abs(model.coef_).sum()
    return loss, time.time() - start


def tests(ns: List[int], ps: List[int], rhos: List[float], L: int) -> pd.DataFrame:
    max_iter = 1000
    tol = 1e-4
    lam = 1
    results = []
    for n, p in tqdm(zip(ns, ps), total=len(ns)):
        for rho in rhos:
            loss_coord, loss_lars, loss_coord_shuffle = [], [], []
            time_coord, time_lars, time_coord_shuffle = [], [], []
            non_zero_coord, non_zero_lars, non_zero_coord_shuffle = [], [], []
            its, convs = [], []
            its_shuffle, convs_shuffle = [], []
            for _ in range(L):
                X, y = generate_data(n, p, rho)
                X = standardize(X)

                model = RegressionModel(lam=lam, tol=tol, max_iter=max_iter)
                start = time.time()
                conv, it = model.fit(X, y, shuffle=False)
                y_pred = model.predict(X)
                loss = (
                    1 / 2 * ((y - y_pred) ** 2).sum()
                    + lam * np.abs(model.weights).sum()
                )
                loss_coord.append(loss)
                time_coord.append(time.time() - start)
                non_zero_coord.append(
                    np.where(np.abs(model.weights) < 1e-12, 0, 1).sum()
                )
                its.append(it)
                convs.append(conv)

                model = RegressionModel(lam=lam, tol=tol, max_iter=max_iter)
                start = time.time()
                conv, it = model.fit(X, y, shuffle=True)
                y_pred = model.predict(X)
                loss = (
                    1 / 2 * ((y - y_pred) ** 2).sum()
                    + lam * np.abs(model.weights).sum()
                )
                loss_coord_shuffle.append(loss)
                time_coord_shuffle.append(time.time() - start)
                non_zero_coord_shuffle.append(
                    np.where(np.abs(model.weights) < 1e-12, 0, 1).sum()
                )
                its_shuffle.append(it)
                convs_shuffle.append(conv)

                model = LassoLars(
                    alpha=lam / len(X), fit_intercept=False, max_iter=max_iter
                )
                start = time.time()
                model.fit(X, y)
                y_pred = model.predict(X)
                loss = ( 
                    1 / 2 * ((y - y_pred) ** 2).sum()
                    + lam * np.abs(model.coef_).sum()
                )
                loss_lars.append(loss)
                time_lars.append(time.time() - start)
                non_zero_lars.append(np.where(np.abs(model.coef_) < 1e-12, 0, 1).sum())

            results.append(
                [
                    "coord",
                    n,
                    p,
                    rho,
                    np.mean(loss_coord),
                    np.mean(time_coord),
                    np.mean(its),
                    np.mean(convs),
                    np.mean(non_zero_coord),
                ]
            )
            results.append(
                [
                    "coord_shuffle",
                    n,
                    p,
                    rho,
                    np.mean(loss_coord_shuffle),
                    np.mean(time_coord_shuffle),
                    np.mean(its_shuffle),
                    np.mean(convs_shuffle),
                    np.mean(non_zero_coord_shuffle),
                ]
            )
            results.append(
                [
                    "LARS",
                    n,
                    p,
                    rho,
                    np.mean(loss_lars),
                    np.mean(time_lars),
                    None,
                    None,
                    np.mean(non_zero_lars),
                ]
            )

    df = pd.DataFrame(
        results,
        columns=[
            "method",
            "n",
            "p",
            "rho",
            "loss",
            "time",
            "it",
            "conv",
            "num_non_zero",
        ],
    )
    return df


if __name__ == "__main__":
    ns = [100, 100, 1000, 5000]
    ps = [1000, 5000, 100, 100]
    rhos = [0, 0.1, 0.2, 0.5, 0.9, 0.95]
    L = 15

    df = tests(ns, ps, rhos, L)
    df.to_csv("results/results.csv", index=False)
