from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.figure as figure
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin


class GTMBase(BaseEstimator, TransformerMixin):
    """
    The original Generative Topographic Mapping (GTM) model by Bishop et al. 1998
    Implemented with reference of GTM toolbox by Lain Strachan written in Matlab.

    Parameters
    ------------
    s : int
        Scale factor of sigma
    l : float
        Regulatory factor. Larger l can prevent singularity during training but
        decrease the model performance and may cause unexpected error.
    map_shape : tuple
        The shape of latent space grid.
    rbf_shape : tuple
        The shape of radial basis function grid.
    pca : bool
        Flag for using PCA to initialize weights W.
    """

    def __init__(self, s=2, l=0.001, map_shape=(16, 16), rbf_shape=(5, 5), pca=True):
        # hyper parameters
        self.l = l  # regularization parameter lambda, larger l can avoid singularity
        self.pca = pca  # PCA initialization flag
        # intermedia parameters
        self.R = None  # responsibility
        self.s = s  # scale parameter of sigma

        # latent point and radial basis function grids
        self.map_shape = map_shape
        self.rbf_shape = rbf_shape
        self.K = np.prod(map_shape)
        self.k = map_shape[0]
        self.M = np.prod(rbf_shape)
        # generate grid coordinates (2d)
        x = np.linspace(-1, 1, self.map_shape[0])
        self.map_grid = np.transpose(np.meshgrid(x, x)).reshape(self.K, 2)
        x = np.linspace(-1, 1, self.rbf_shape[0])
        self.rbf_grid = np.transpose(np.meshgrid(x, x)).reshape(self.M, 2)
        self.rbf_grid = self.rbf_grid * (rbf_shape[0] / (rbf_shape[0] - 1))
        #  sigma equals the distance between two neighbour of rbf centers * s
        self.sigma = abs(self.rbf_grid[0, 1] - self.rbf_grid[1, 1]) * self.s
        # init phi
        dist = cdist(self.map_grid, self.rbf_grid, metric='sqeuclidean')
        # phi: K * M+1 (plus bias)
        self.phi: np.ndarray = np.exp(-dist / (2 * self.sigma ** 2))
        self.phi_with_ones = np.c_[self.phi, np.ones([self.phi.shape[0], 1])]

        # data specified parameters
        self.N = None  # num of observations
        self.D = None  # dimension
        self.T = None  # data
        self.means = None  # data means
        self.W = None  # weight matrix
        self.beta = None  # variance coef
        self.mu = None  # projection in data space

        # training history
        self.llhs = []

    def fit(self, x, y=None, epoch=10, early_stopping=False, tol=10, verbose=True):
        """
        train the model. y will not be used.
        """
        self._init_vars(x)
        self.llhs = []
        for epoch in range(epoch):
            llh = self._optimize()
            self.llhs.append(llh)
            # early stopping when loglikelihood converged
            if epoch >= 2 and early_stopping and abs(self.llhs[-2] - llh) < tol:
                print("Early stopping since variance of llh is less than tol")
                self.calibrate()
                break
            if verbose:
                print(f"{epoch}: {llh}")
        return self

    def transform(self, x, y=None, mode='mode') -> np.ndarray:
        r"""Project the dataset into latent space.

        Parameters
        -------------
        x : np.ndarray
            data vec with shape (n_obs, dim)
        y : None
            placeholder
        mode : str
            the posterior mode. It should be either 'mode' or 'mean'

        Returns
        -------------
        np.ndarray
            coors of points in shape (n, 2)

        """
        mu = self.phi_with_ones @ self.W
        dist = cdist(x, mu, metric='sqeuclidean')
        exp_term: np.ndarray = np.exp((-self.beta / 2) * dist)
        # sum_k(p(tn|xk, W, beta))  (size=n)
        exp_term_sum_over_k = exp_term.sum(axis=1)
        # R: responsibility matrix where rkn = p(tn|xn, W, beta)/sum_k((p(tn|xk, W, beta)))
        # R: K * N array
        responsibility: np.ndarray = (exp_term / exp_term_sum_over_k.reshape(-1, 1)).T

        if mode == 'mode':
            modes = self.map_grid[responsibility.argmax(axis=0)]
            return modes
        elif mode == 'mean':
            means = self.map_grid[responsibility.T.dot(self.map_grid)]
            return means
        else:
            raise ValueError(f"Error with parameter mode, expect mean or mode, got {mode}")

    def calibrate(self):
        """ calibrate the R matrix after last optimize """
        dist = cdist(self.T, self.phi_with_ones.dot(self.W), metric='sqeuclidean')
        exp_term: np.ndarray = np.exp((-self.beta / 2) * dist)
        exp_term_sum_over_k = exp_term.sum(axis=1)
        self.R: np.ndarray = (exp_term / exp_term_sum_over_k.reshape(-1, 1)).T

    def plot_llh(self):
        """Plot the training log-likelihood figure"""
        plt.plot(self.llhs)
        plt.title("Training Log-likelihood")
        plt.xlabel("epoch")
        plt.ylabel("llh")
        plt.show()

    def score(self, x):
        """score the model performance using current log-likelihood"""
        dist = cdist(self.T, self.mu, metric='sqeuclidean')
        exp_term: np.ndarray = np.exp((-self.beta / 2) * dist)
        llh = np.log((self.beta / (2 * np.pi)) ** (self.D / 2) * exp_term.sum(axis=1) / self.K).sum()
        return llh

    def plot(self, mode='mode', label: None | np.ndarray = None, num_points=200):
        """
        Label is for every point and in shape of (n_points, )
        You might want to call np.repeat(label, n) to repeat every label n times
        """
        plt.figure(figsize=figure.figaspect(1))
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        if mode == 'mode':
            modes = self.map_grid[self.R.argmax(axis=0)][:num_points]
            plt.scatter(modes[:, 0], modes[:, 1], c=label)
            plt.xlabel("z1 (mode)")
            plt.ylabel("z2 (mode)")
            plt.show()
        elif mode == 'mean':
            means = self.R.T.dot(self.map_grid)[:num_points]
            plt.scatter(means[:, 0], means[:, 1], c=label)
            plt.xlabel("z1 (mean)")
            plt.ylabel("z2 (mean)")
            plt.show()

    def _init_vars(self, x: np.ndarray):
        r"""Initialize the hyperparameters related to training data.

        Parameters
        =============
        x : np.ndarray
            training data in shape (n_obs, dim)

        """
        # data related fixed parameters
        self.T: np.ndarray = x
        self.N = self.T.shape[0]
        self.D = self.T.shape[1]

        # init bias
        self.means: np.ndarray = x.mean(axis=0)
        # init W and beta
        pca = PCA(n_components=3)
        pca.fit(x)
        # scale eigen vectors based on their eigen values
        tmp = (pca.components_.T @ np.diag(np.sqrt(pca.explained_variance_)))[:, :2]
        # normalize map grid base on the original code.
        # not sure why it has to be done
        # this step generate different result comparing to Matlab impl because of higher accuracy.
        norm_map_grid = (self.map_grid - np.ones(self.map_grid.shape).dot(np.diag(self.map_grid.mean(axis=0)))).dot(
            np.diag(1 / np.std(self.map_grid, axis=0)))
        # self.W = np.linalg.pinv(self.phi_with_ones) @ norm_map_grid @ tmp.T
        self.W = np.linalg.pinv(self.phi_with_ones).dot(norm_map_grid.dot(tmp.T))
        self.W[-1] = self.means
        # (L+1)th principal component
        p1 = pca.explained_variance_[2]
        # half the average minimal distance between the mixture components
        # p2 = euclidean_distances(self.phi.dot(self.W) + self.bias)  # pairwise distance
        p2 = euclidean_distances(self.phi_with_ones.dot(self.W))  # pairwise distance
        np.fill_diagonal(p2, np.inf)  # fill the diag since the dist to self is 0
        p2 = p2.min(axis=0).mean() / 2
        beta_inv = max(p1, p2)
        self.beta = 1 / beta_inv
        self.mu = self.phi_with_ones.dot(self.W)

    def _optimize(self) -> float:
        """A single optimizing step with the EM algorithm"""
        # E-steps
        # dist: N * K array
        dist = cdist(self.T, self.mu, metric='sqeuclidean')
        # exp{(-beta/2) * dist}
        exp_term: np.ndarray = np.exp((-self.beta / 2) * dist)
        # sum_k(p(tn|xk, W, beta))  (size=n)
        exp_term_sum_over_k = exp_term.sum(axis=1)
        # R: responsibility matrix where rkn = p(tn|xn, W, beta)/sum_k((p(tn|xk, W, beta)))
        # R: K * N array
        self.R: np.ndarray = (exp_term / exp_term_sum_over_k.reshape(-1, 1)).T
        # R sum over n (size=k)
        # G: where g_kk = sum_n(r_kn)
        G: np.ndarray = np.diag(self.R.sum(axis=1))
        # calculate phi^T . G . phi
        phi_t_g_phi = self.phi_with_ones.T.dot(G).dot(self.phi_with_ones)

        # log likelihood p(t|x)
        llh = np.log((self.beta / (2 * np.pi)) ** (self.D / 2) * exp_term_sum_over_k / self.K).sum()

        # M-step
        regul = np.identity(phi_t_g_phi.shape[0]) * self.l
        regul[-1, -1] = 0
        self.W = np.linalg.pinv(phi_t_g_phi + regul) \
            .dot(self.phi_with_ones.T).dot(self.R).dot(self.T)
        dist = cdist(self.T, self.phi_with_ones.dot(self.W), metric='sqeuclidean')
        self.beta = (self.N * self.D) / (np.multiply(self.R.T, dist)).sum()

        # update projection
        self.mu = self.phi_with_ones.dot(self.W)

        return llh
