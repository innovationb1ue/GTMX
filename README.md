# GTMX
## Description
A Python package for Generative Topographic Mapping (GTM)  
Provide original version of GTM by Bishop et al. (1998) and GTM through time by Bishop et al. (1997). 

Link to the documentation: https://gtmx.readthedocs.io/en/latest/

## Installation

### pip
`pip install gtmx`  
### source  
Clone the repository and cd into project root. 
`pip install .`

## Example
for basic GTM model
```python
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
```
for GTM through time
```python
from gtmx import GTMTimeSeries
import numpy as np

x = np.ndarray(["Your time series data in shape (n_obs, sequence, dimension)"])

gtm = GTMTimeSeries()
gtm.fit(x)
gtm.plot_llh()
gtm.plot('mean')
gtm.plot('mode')

```

## Main References
Nabney, I. (2002). NETLAB: algorithms for pattern recognition. Springer Science & Business Media.  
Bishop, C. M., Svensén, M., & Williams, C. K. (1998). GTM: The generative topographic mapping. Neural computation, 10(1), 215-234.  
Bishop, C. M., Hinton, G. E., & Strachan, I. G. (1997). GTM through time.
Gaspar, H. A. (2018). ugtm: A Python package for data modeling and visualization using generative topographic mapping. Journal of Open Research Software, 6(1).




