from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from scipy.optimize import linear_sum_assignment


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_train, y_train, x_test, y_test


def get_callbacks(patience: int, path: str) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="../models/" + path,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
        ),
    ]


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true[y_pred >= 0]
    y_pred = y_pred[y_pred >= 0]
    dim = max(y_pred.max(), y_true.max()) + 1
    cost_mat = np.zeros((dim, dim), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_mat[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(cost_mat.max() - cost_mat)

    return sum([cost_mat[i, j] for i, j in zip(*ind)]) * 1.0 / np.sum(cost_mat)