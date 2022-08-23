# build fixtures using pytest.fixture decorator if necessary
from sklearn.datasets import load_iris
import numpy as np
from src.gtmx import GTMBase
from src.gtmx import GTMTimeSeries
import pytest
import pandas as pd


@pytest.fixture()
def original_gtm():
    iris = load_iris()
    X: np.ndarray = iris.data
    Y = iris.target
    e = GTMBase(map_shape=(14, 14), rbf_shape=(4, 4), s=2, l=0.01)
    e.fit(X, Y, epoch=50)
    return e


@pytest.fixture()
def gtmtt():
    iris = load_iris()
    X: np.ndarray = iris.data
    Y = iris.target
    X = X.reshape([1, X.shape[0], X.shape[1]])
    e = GTMTimeSeries(map_shape=(14, 14), rbf_shape=(4, 4), s=2, l=0.01, group_size=2)
    e.fit(X, Y, epoch=50)
    return e


@pytest.fixture()
def example_data():
    dta = pd.read_excel("../data/example_data.xlsx", sheet_name=0, header=None)
    dta = dta.to_numpy()
    dta = dta[:, :9]  # first 9 cols
    print(dta.shape)
    dta = dta.reshape([4, dta.shape[0] // 4, 9])
    return dta
