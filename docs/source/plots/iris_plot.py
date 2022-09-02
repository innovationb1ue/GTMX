import matplotlib.pyplot as plt
from gtmx import GTMBase
from sklearn.datasets import load_iris


iris = load_iris()
x = iris.data
y = iris.target

gtm = GTMBase(l=1)
gtm.fit(x, epoch=30)
gtm.plot_llh()
gtm.plot('mean', label=y)
gtm.plot('mode', label=y)

plt.show()