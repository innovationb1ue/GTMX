"""
Artificial S-curve visualization
====================================
"""
import pandas as pd
from sklearn import datasets
from sklearn import manifold
from gtmx import GTMBase

import matplotlib.pyplot as plt


X,y = datasets.make_s_curve(n_samples=1000, random_state=0)
man = manifold.TSNE(n_components=2, init='pca', random_state=0)
tsne = man.fit_transform(X)
man = manifold.MDS(max_iter=100, n_init=1, random_state=0)
mds = man.fit_transform(X)
man = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=2,
                                      eigen_solver='auto',
                                      method="standard",
                                      random_state=0)
lle = man.fit_transform(X)

# Construct GTM


gtm = GTMBase(s=0.3, map_shape=(12, 12), rbf_shape=(2, 2), l=0.01)

gtm_means = gtm.fit_transform(X)
gtm_modes = gtm.fit_transform(X)

dgtm_modes = pd.DataFrame(gtm_modes, columns=["x1", "x2"])
dgtm_modes["label"] = y


fig, axs = plt.subplots(2, 2, constrained_layout=True)


axs[0, 0].scatter(dgtm_modes['x1'], dgtm_modes['x2'])
axs[0, 0].set_title('gtm modes')

dgtm_means = pd.DataFrame(gtm_means, columns=["x1", "x2"])
dgtm_means["label"] = y

axs[0, 1].scatter(dgtm_means['x1'], dgtm_means['x2'])
axs[0, 1].set_title('gtm means')

dtsne = pd.DataFrame(tsne, columns=["x1", "x2"])
dmds = pd.DataFrame(mds, columns=["x1", "x2"])
dlle = pd.DataFrame(lle, columns=["x1", "x2"])
dtsne["label"] = y
dmds["label"] = y
dlle["label"] = y


axs[1, 0].scatter(dtsne['x1'], dtsne['x2'])
axs[1, 0].set_title('t-SNE')

axs[1, 1].scatter(dmds['x1'], dmds['x2'])
axs[1, 1].set_title('MDS')


plt.show()



