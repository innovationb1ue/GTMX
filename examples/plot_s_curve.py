"""
Artificial S-curve visualization
====================================
"""
import altair as alt
import pandas as pd
from sklearn import datasets
from sklearn import manifold
from gtmx import GTMBase


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

gtm_modes = alt.Chart(dgtm_modes).mark_circle().encode(
    x='x1',
    y='x2',
    color=alt.Color('label:Q',
                    scale=alt.Scale(scheme='viridis')),
    size=alt.value(50),
    tooltip=['x1','x2','label:Q']
).properties(title = "GTM (modes)", width = 100, height = 100)

dgtm_means = pd.DataFrame(gtm_means, columns=["x1", "x2"])
dgtm_means["label"] = y

gtm_means = alt.Chart(dgtm_means).mark_circle().encode(
    x='x1',
    y='x2',
    color=alt.Color('label:Q',
                    scale=alt.Scale(scheme='viridis')),
    size=alt.value(50),
    tooltip=['x1','x2','label:Q']
).properties(title = "GTM (means)", width = 100, height = 100)

dtsne = pd.DataFrame(tsne, columns=["x1", "x2"])
dmds = pd.DataFrame(mds, columns=["x1", "x2"])
dlle = pd.DataFrame(lle, columns=["x1", "x2"])
dtsne["label"] = y
dmds["label"] = y
dlle["label"] = y

tsne = alt.Chart(dtsne).mark_circle().encode(
    x='x1',
    y='x2',
    color=alt.Color('label:Q',
                    scale=alt.Scale(scheme='viridis')),
    size=alt.value(50),
    tooltip=['x1','x2','label:Q']
).properties(title = "t-SNE", width = 100, height = 100)

mds = alt.Chart(dmds).mark_circle().encode(
    x='x1',
    y='x2',
    color=alt.Color('label:Q',
                    scale=alt.Scale(scheme='viridis')),
    size=alt.value(50),
    tooltip=['x1','x2','label:Q']
).properties(title = "MDS", width = 100, height = 100)

gtm = gtm_means | gtm_modes
others = tsne | mds

alt.vconcat(gtm, others)