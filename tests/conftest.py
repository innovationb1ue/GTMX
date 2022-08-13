# build fixtures using pytest.fixture decorator if necessary
from sklearn.datasets import load_iris
import numpy as np
from src.gtmx import GTMBase
import pytest


@pytest.fixture()
def original_gtm():
    iris = load_iris()
    X: np.ndarray = iris.data
    Y = iris.target
    e = GTMBase(map_shape=(14, 14), rbf_shape=(4, 4), s=2, l=0.01)
    e.fit(X, Y, epoch=50)
    return e



