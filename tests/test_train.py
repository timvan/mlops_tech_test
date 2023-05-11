import os

import numpy as np
import pandas as pd
import pytest

from src.train import DataHandler


@pytest.fixture
def test_data():
    num_rows = 1000
    random_classes = np.random.rand(num_rows)
    random_features = np.random.rand(num_rows)
    df = pd.DataFrame({"target": random_classes, "feature": random_features})
    file_path = "tests/test_data.csv"
    df.to_csv(file_path)

    yield file_path

    os.remove(file_path)


def test_data_handler(test_data):

    data_handler = DataHandler(test_data, "target")
    data_handler.load_data()
    X_train, X_test, X_val, y_train, y_test, y_val = data_handler.split_data(0.4, 0.1)

    count_rows = X_train.shape[0] + X_test.shape[0] + X_val.shape[0]

    assert round(X_train.shape[0] / count_rows, 1) == 0.5
    assert round(X_test.shape[0] / count_rows, 1) == 0.4
    assert round(X_val.shape[0] / count_rows, 1) == 0.1


def test_data_handler_throws_exception():

    data_handler = DataHandler("./doesnotexist", "target")

    with pytest.raises(FileNotFoundError):
        data_handler.load_data()
