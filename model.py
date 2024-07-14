import numpy as np


class KNNModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((x - self.X_train)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            y_pred[i] = np.argmax(np.bincount(k_labels))
        return y_pred
