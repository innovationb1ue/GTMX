from sklearn.datasets import load_iris
import numpy as np
from src.gtmx import GTMBase


def test_gtm_base():
    iris = load_iris()
    X: np.ndarray = iris.data
    Y = iris.target
    e = GTMBase(map_shape=(14, 14), rbf_shape=(4, 4), s=2, l=0.01)
    e.fit(X, Y, epoch=50)
    e.plot_llh()
    e.plot(label=Y)
    e.plot(mode='mean', label=Y)
