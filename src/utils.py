
import os
import numpy as np


def load_numpy_data(file_path):
    #Load a saved NumPy dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return np.load(file_path)


def print_data_shapes(X_train, X_test, y_train, y_test):
    #Display training and testing dataset shapes
    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
