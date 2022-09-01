GTMTT Examples
=================

Custom Dataset
-------------------

.. code-block:: python

    from gtmx import GTMTimeSeries
    import numpy as np

    x = np.ndarray(["Your time series data in shape (n_obs, sequence, dimension)"])

    gtm = GTMTimeSeries()
    gtm.fit(x)
    gtm.plot_llh()
    gtm.plot('mean')
    gtm.plot('mode')