import numpy as np
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from model import KNNModel


class ArgumentStorage:
    def __init__(self, args: dict):
        self.__dict__.update(args)


def load_data(
        test_ratio: 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    X: np.ndarray = iris.data[:, 1:3]
    y: np.ndarray = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=2023)
    return X_train, X_test, y_train, y_test


def plot_decision_boundary(
        model: KNNModel,
        X: np.ndarray,
        y: np.ndarray,
        partition: str = "train",
) -> None:
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title("KNN (k=%d): %s partition" % (model.k, partition))
    plt.show()
