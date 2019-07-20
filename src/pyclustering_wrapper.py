from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.ema import ema
from pyclustering.cluster.bsas import bsas
from pyclustering.cluster.mbsas import mbsas
from pyclustering.cluster.ttsas import ttsas
from pyclustering.cluster.rock import rock
from pyclustering.cluster.somsc import somsc
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import distance_metric, metric
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from scipy.stats import randint
import numpy as np
import random


class SOMSCWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=5, epoch=100):
        self.n_clusters = n_clusters
        self.epoch = epoch
        self.wrapped_instance = None
        self.labels_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = somsc(data=x,
                                      amount_clusters=self.n_clusters,
                                      epouch=self.epoch
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = somsc(data=x,
                                      amount_clusters=self.n_clusters,
                                      epouch=self.epoch
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()

        return self


class RockWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=5, eps=1.0, threshold=0.5):
        self.n_clusters = n_clusters
        self.eps = eps
        self.threshold = threshold
        self.wrapped_instance = None
        self.labels_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = rock(data=x,
                                     eps=self.eps,
                                     number_clusters=self.n_clusters,
                                     threshold=self.threshold
                                     )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = rock(data=x,
                                     eps=self.eps,
                                     number_clusters=self.n_clusters,
                                     threshold=self.threshold
                                     )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()

        return self


class TTSASWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, max_n_clusters=5, threshold_1=0.3, threshold_2=0.8):
        self.max_n_clusters = max_n_clusters
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.metric = distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.representatives_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = ttsas(data=x,
                                      maximum_clusters=self.max_n_clusters,
                                      threshold1=self.threshold_1,
                                      threshold2=self.threshold_2,
                                      metric=self.metric,
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = ttsas(data=x,
                                      maximum_clusters=self.max_n_clusters,
                                      threshold1=self.threshold_1,
                                      threshold2=self.threshold_2,
                                      metric=self.metric,
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()


class MBSASWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, max_n_clusters=5, threshold=0.5):
        self.max_n_clusters = max_n_clusters
        self.threshold = threshold
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.representatives_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = mbsas(data=x,
                                      maximum_clusters=self.max_n_clusters,
                                      threshold=self.threshold,
                                      metric=self.metric,
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = mbsas(data=x,
                                      maximum_clusters=self.max_n_clusters,
                                      threshold=self.threshold,
                                      metric=self.metric,
                                      )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()

        return self


class BSASWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, max_n_clusters=5, threshold=0.5):
        self.max_n_clusters = max_n_clusters
        self.threshold = threshold
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.representatives_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = bsas(data=x,
                                     maximum_clusters=self.max_n_clusters,
                                     threshold=self.threshold,
                                     metric=self.metric,
                                     )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')

        self.wrapped_instance = bsas(data=x,
                                     maximum_clusters=self.max_n_clusters,
                                     threshold=self.threshold,
                                     metric=self.metric,
                                     )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.representatives_ = self.wrapped_instance.get_representatives()

        return self


class ExpectationMaximizationWrapper(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=5, tolerance=0.00001, max_iter=100):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.centers_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')
        means, covs = self.__initial_mean_cov(x)

        self.wrapped_instance = ema(data=x,
                                    amount_clusters=self.n_clusters,
                                    means=means,
                                    variances=covs,
                                    tolerance=self.tolerance,
                                    )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.centers_ = self.wrapped_instance.get_centers()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')
        means, covs = self.__initial_mean_cov(x)

        self.wrapped_instance = ema(data=x,
                                    amount_clusters=self.n_clusters,
                                    means=means,
                                    variances=covs,
                                    tolerance=self.tolerance,
                                    )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.centers_ = self.wrapped_instance.get_centers()

        return self

    def __initial_mean_cov(self, x):
        kmeans_instance = kmeans(data=x,
                                 initial_centers=kmeans_plusplus_initializer(x, self.n_clusters).initialize(),
                                 metric=self.metric,
                                 itermax=self.max_iter)
        kmeans_instance.process()

        means = kmeans_instance.get_centers()

        covariances = []
        initial_clusters = kmeans_instance.get_clusters()
        for initial_cluster in initial_clusters:
            if len(initial_cluster) > 1:
                cluster_sample = [x[index_point] for index_point in initial_cluster]
                covariances.append(np.cov(cluster_sample, rowvar=False))
            else:
                dimension = len(x[0])
                covariances.append(np.zeros((dimension, dimension)) + random.random() / 10.0)

        return means, covariances


class KMediansWrapper(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=5, tolerance=0.001):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.medians_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmedians(data=x,
                                         initial_centers=kmeans_plusplus_initializer(x, self.n_clusters).initialize(),
                                         tolerance=self.tolerance,
                                         metric=self.metric
                                         )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.medians_ = self.wrapped_instance.get_medians()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmedians(data=x,
                                         initial_centers=kmeans_plusplus_initializer(x, self.n_clusters).initialize(),
                                         tolerance=self.tolerance,
                                         metric=self.metric
                                         )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.medians_ = self.wrapped_instance.get_medians()

        return self


class KMedoidsWrapper(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=5, tolerance=0.001):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.wrapped_instance = None
        self.labels_ = None
        self.medoids_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmedoids(data=x,
                                         initial_index_medoids=[randint(0, x.shape[0]) for _ in range(0, x.shape[0])],
                                         tolerance=self.tolerance,
                                         metric=self.metric
                                         )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.medoids_ = self.wrapped_instance.get_medoids()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmedoids(data=x,
                                         initial_index_medoids=[randint(0, x.shape[0]) for _ in range(0, x.shape[0])],
                                         tolerance=self.tolerance,
                                         metric=self.metric
                                         )
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.medoids_ = self.wrapped_instance.get_medoids()

        return self


class KMeansWrapper(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=5, tolerance=0.001, max_iter=200):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.metric = metric.distance_metric(metric.type_metric.MANHATTAN)
        self.max_iter = max_iter
        self.wrapped_instance = None
        self.labels_ = None
        self.centers_ = None

    def fit_predict(self, x, y=None):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmeans(data=x,
                                       initial_centers=kmeans_plusplus_initializer(x, self.n_clusters).initialize(),
                                       tolerance=self.tolerance,
                                       metric=self.metric,
                                       itermax=self.max_iter)
        self.wrapped_instance.process()
        self.labels_ = self.wrapped_instance.get_clusters()
        self.centers_ = self.wrapped_instance.get_centers()

        return self.labels_

    def fit(self, x):
        x = check_array(x, accept_sparse='csr')
        self.wrapped_instance = kmeans(data=x,
                                       initial_centers=kmeans_plusplus_initializer(x, self.n_clusters).initialize(),
                                       tolerance=self.tolerance,
                                       metric=self.metric,
                                       itermax=self.max_iter)
        self.wrapped_instance.process()

        self.labels_ = self.wrapped_instance.get_clusters()
        self.centers_ = self.wrapped_instance.get_centers()

        return self
