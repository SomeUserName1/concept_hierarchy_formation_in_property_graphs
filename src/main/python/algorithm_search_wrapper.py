from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from pyclustering_wrapper import KMeansWrapper, KMediansWrapper, KMedoidsWrapper, BSASWrapper, MBSASWrapper, \
    TTSASWrapper, RockWrapper, SOMSCWrapper
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, SpectralClustering
from sklearn import metrics
from scipy.stats import randint, uniform


def pyclust_params(n_samples):
    return {
            "n_clusters": randint(4, (n_samples / 2) + 4),
            "tolerance": uniform(0.00001, 0.1)
    }


def cluster_affinity_prop():
    grid_params = {
        "damping": uniform(loc=0.5, scale=0.5)
    }
    clust = AffinityPropagation(affinity='precomputed')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)
    return searcher


def cluster_spectral(n_samples):
    grid_params = {
        "n_clusters": randint(5, n_samples / 2 + 5),
    }
    clust = SpectralClustering(affinity='precomputed', eigen_solver='arpack', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)
    return searcher


def cluster_dbscan(n_samples):
    grid_params = {
        "eps": uniform(loc=0.1, scale=5.0),
        "min_samples": uniform(0, 0.1),
        "leaf_size": randint(5, round(0.1 * n_samples) + 5)
    }
    clust = DBSCAN(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, error_score='raise', refit=True, n_iter=100)
    return searcher


def cluster_optics(n_samples):
    grid_params = {
        "min_samples": uniform(0.000001, 0.3),
        "leaf_size": randint(5, round(0.1 * n_samples) + 5),
        "min_cluster_size": uniform(0.0000001, 0.1)
    }
    clust = OPTICS(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


def cluster_birch():
    grid_params = {
        "threshold": uniform(loc=0.1, scale=0.9),
        "branching_factor": randint(5, 100),
    }
    clust = Birch(n_clusters=None)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


def cluster_kmeans(n_samples):
    return RandomizedSearchCV(KMeansWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)


def cluster_kmedoids(n_samples):
    return RandomizedSearchCV(KMedoidsWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)


def cluster_kmedians(n_samples):
    return RandomizedSearchCV(KMediansWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)


def cluster_bsas(n_samples):
    params =  {
            "max_n_clusters": randint(4, (n_samples / 2) + 4),
            "threshold": uniform(0.00001, 0.1)
    }
    return RandomizedSearchCV(BSASWrapper(), param_distributions=params, cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)


def cluster_mbsas(n_samples):
    grid_params = {
        "max_n_clusters": randint(4, (n_samples / 2) + 4),
        "threshold": uniform(0, 1)
    }
    clust = MBSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


def cluster_ttsas():
    grid_params = {
        "threshold_1": uniform(loc=0, scale=0.5),
        "threshold_2": uniform(0.5, 1)
    }
    clust = TTSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


def cluster_rock(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "eps": uniform(0.0, 1),
        "threshold": uniform(0.0, 1)
    }
    clust = RockWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


def cluster_som(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "epoch": randint(10, 300)
    }
    clust = SOMSCWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=100)

    return searcher


# _________________ Parameter Search Util
def cv_scorer(estimator, x):
    estimator.fit(x)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(x)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(x, cluster_labels) * metrics.calinski_harabasz_score(x, cluster_labels)


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    @staticmethod
    def split(x, _, __):
        yield np.arange(len(x)), np.arange(len(x))

    def get_n_splits(self, _, __, ___):
        return self.n_splits
