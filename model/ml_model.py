import numpy as np
from sklearn.linear_model import LinearRegression


def train_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = LinearRegression().fit(X, y)
    return model


if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully!")
