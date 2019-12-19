from hdbscan import RobustSingleLinkage, HDBSCAN
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from pyclustering_wrapper import KMeansWrapper, KMediansWrapper, KMedoidsWrapper, BSASWrapper, MBSASWrapper, \
    TTSASWrapper, RockWrapper, SOMSCWrapper
from sklearn.cluster import DBSCAN, AffinityPropagation, OPTICS, Birch, SpectralClustering
from sklearn import metrics
from scipy.stats import randint, uniform


def pyclust_params(n_samples):
    return {
            "n_clusters": randint(2, (n_samples / 100) + 8),
            "tolerance": uniform(0.00001, 0.1)
    }


def cluster_rsl(n_samples):
    grid_params = {
        "k": randint(2, (n_samples / 100) + 8),
        "alpha": uniform(1, 0.5),
        "gamma": randint(2, (n_samples / 100) + 8),
        "cut": uniform(0.2, 0.8)
    }
    clust = RobustSingleLinkage(metric='jaccard', algorithm='boruvka_balltree')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=3)
    return searcher


def cluster_hdbscan(n_samples):
    grid_params = {
        "alpha": uniform(1, 0.5),
        "leaf_size": randint(2, n_samples / 4 + 100)
    }
    clust = HDBSCAN(metric='jaccard', core_dist_n_jobs=-1, algorithm='boruvka_balltree')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=3)
    return searcher


def cluster_dbscan(n_samples):
    grid_params = {
        "eps": uniform(loc=0.5, scale=0.5),
        "leaf_size": randint(2, n_samples/4 + 100)
    }
    clust = DBSCAN(metric='jaccard', n_jobs=-1, algorithm='ball_tree')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, error_score='raise', refit=True, n_iter=3)
    return searcher


def cluster_optics(n_samples):
    grid_params = {
        "leaf_size": randint(20, n_samples/4 + 100),
    }
    clust = OPTICS(metric='jaccard', algorithm='ball_tree', max_eps=0.5)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=3)

    return searcher


def cluster_ttsas():
    grid_params = {
        "threshold_1": uniform(loc=0.7, scale=0.05),
        "threshold_2": uniform(0.75, 0.05)
    }
    clust = TTSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)

    return searcher


def cluster_affinity_prop():
    grid_params = {
        "damping": uniform(loc=0.5, scale=0.5)
    }
    clust = AffinityPropagation(affinity='jaccard')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)
    return searcher


def cluster_spectral(n_samples):
    grid_params = {
     "n_clusters": randint(2, n_samples / 2 + 2),
    }
    clust = SpectralClustering(affinity='precomputed', eigen_solver='arpack', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                               n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)
    return searcher


def cluster_birch():
    grid_params = {
        "threshold": uniform(loc=0.1, scale=0.9),
        "branching_factor": randint(2, 50),
    }
    clust = Birch(n_clusters=None)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)

    return searcher


def cluster_kmeans(n_samples):
    return RandomizedSearchCV(KMeansWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)


def cluster_kmedoids(n_samples):
    return RandomizedSearchCV(KMedoidsWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)


def cluster_kmedians(n_samples):
    return RandomizedSearchCV(KMediansWrapper(), param_distributions=pyclust_params(n_samples), cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)


def cluster_bsas(n_samples):
    params = {
            "max_n_clusters": randint(4, (n_samples / 2) + 4),
            "threshold": uniform(0.00001, 0.1)
    }
    return RandomizedSearchCV(BSASWrapper(), param_distributions=params, cv=DisabledCV(),
                              error_score='raise', n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)


def cluster_mbsas(n_samples):
    grid_params = {
        "max_n_clusters": randint(4, (n_samples / 2) + 4),
        "threshold": uniform(0, 1)
    }
    clust = MBSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)

    return searcher


def cluster_rock(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "eps": uniform(0.0, 1),
        "threshold": uniform(0.0, 1)
    }
    clust = RockWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=5)

    return searcher


def cluster_som(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "epoch": randint(10, 300)
    }
    clust = SOMSCWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=20)

    return searcher


# _________________ Parameter Search Util
def cv_scorer(estimator, x):
    x = np.array(x).astype(bool)
    estimator.fit(x)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(x)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(x, cluster_labels, metric='jaccard') \
               * metrics.calinski_harabasz_score(x, cluster_labels)


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    @staticmethod
    def split(x, _, __):
        yield np.arange(len(x)), np.arange(len(x))

    def get_n_splits(self, _, __, ___):
        return self.n_splits
