from pyclustering.cluster.kmeans import kmeans
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np


class KMeans(BaseEstimator, ClusterMixin):

    def __init__(self, data, initial_centers, tolerance=0.001, max_iter=200):
        self.data = data
        self.initial_centers = initial_centers
        self.tolerance = tolerance
        self.metric = self.__jaccard
        self.max_iter = max_iter
        self.wrapped_instance = kmeans(self.data, self.initial_centers, self.tolerance, self.metric)

    def fit_predict(self, X, y=None):
        self.data = X
        self.wrapped_instance.process()
        return self.wrapped_instance.get_clusters()

    def __jaccard(p1, p2):
        return (np.logical_and(p1, p2).sum() / float(np.logical_or(p1, p2).sum())).T