import json
import logging
import math
import random
import time
from enum import Enum
from shlex import split
from subprocess import Popen, PIPE
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
from line_profiler import LineProfiler
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import randint, uniform
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation, SpectralClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

from src.pyclustering_wrapper import KMeansWrapper, KMediansWrapper, KMedoidsWrapper, ExpectationMaximizationWrapper, \
    BSASWrapper, MBSASWrapper, TTSASWrapper, RockWrapper, SOMSCWrapper

from hdbscan import HDBSCAN, RobustSingleLinkage, condense_tree, label

BASE = "/home/fabian/Nextcloud/workspace/uni/8/bachelor_project"
CACHE_PATH = "/tmp/"


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield np.arange(len(X)), np.arange(len(X))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# Restructure into pipeline:


#   4. Cluster data:
#                    TODO apply Robust single linkage on top of all: what are the distances between clusters
#                    TODO add conceptual clustering
#
#   5. Evaluate     TODO create bracket tree from robust single linkage output
#   6. Visualize


class Dataset(Enum):
    SYNTHETIC = (BASE + "/data/synthetic.json", BASE + "/data/synthetic.tree")
    NOISY_SYNTHETIC = (BASE + "/data/synthetic_noisy.json", BASE + "/data/synthetic.tree")
    YELP = (BASE + "/data/business.json", BASE + "/data/business.tree")


def generate_synthetic(depth: int, width: int = 2, path: str = BASE + "/data/", rem_labels: int = 1,
                       add_labels: int = 0, alter_labels: int = 1, prob: float = 0.33):
    command = "java -jar " + BASE + "/lib/synthetic_data_generator.jar -p '" + path + "' -d " + str(
        depth) + " -w " + str(width) \
              + " -n " + str(rem_labels) + " " + str(add_labels) + " " + str(alter_labels) + " -pr " + str(prob)
    args = split(command)
    Popen(args).wait()


def sample_yelp(n_samples: int) -> List[List[str]]:
    sample = []
    with open(Dataset.YELP.value[0], "r") as read_file:
        for i, line in enumerate(read_file):
            if i < n_samples:
                sample.append(json.loads(line))
            elif i >= n_samples and random.random() < n_samples / float(i + 1):
                replace = random.randint(0, len(sample) - 1)
                sample[replace] = json.loads(line)

        labels = [entry['categories'] for entry in sample if entry is not None]

    return [labels]


def open_synthetic(noisy: bool):
    path = Dataset.SYNTHETIC.value[0] if noisy else Dataset.NOISY_SYNTHETIC.value[0]
    with open(path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]

    return labels


def load(n_samples: int, dataset: Dataset) -> List[List[str]]:
    if dataset == Dataset.SYNTHETIC or Dataset.NOISY_SYNTHETIC:
        generate_synthetic(int(math.log2(n_samples)))
        noisy = True if dataset == Dataset.NOISY_SYNTHETIC else False
        return open_synthetic(noisy)
    else:
        return sample_yelp(n_samples)


def cluster_robust_single_linkage():
    grid_params = {
        "cut": triangular_generator(0.1, 1.0, 0.4),
        "k": triangular_generator(1, 100, 5),
        "alpha": triangular_generator(0.1, 5, 1.4142),
        "gamma": triangular_generator(1, 30, 5),
    }
    clust = RobustSingleLinkage(metric='jaccard', core_dist_n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1,
                                  scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_agglomerative():
    return AgglomerativeClustering(n_clusters=2, affinity='jaccard', memory=CACHE_PATH, linkage='complete')


def cluster_affinity_prop():
    grid_params = {
        "damping": triangular_generator(0.5, 1.0, 0.7)
    }
    clust = AffinityPropagation(affinity='precomputed')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)
    return searcher


def cluster_spectral():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
    }
    clust = SpectralClustering(affinity='precomputed', eigen_solver='amg', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)
    return searcher


def cluster_dbscan():
    grid_params = {
        "eps": uniform(),
        "min_samples": randint(2, 8),
        "leaf_size": randint(3, 10)
    }
    clust = DBSCAN(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, error_score='raise', refit=True, n_iter=100)
    return searcher


def cluster_optics():
    grid_params = {
        "max_eps": randint(0.1, 1),
        "min_samples": randint(0.01, 1),
        "leaf_size": randint(5, 100),
        "xi": triangular_generator(0, 1, 0.05),
        "min_cluster_size": triangular_generator(0, 1, 0.2)
    }
    clust = OPTICS(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmeans():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "tolerance": triangular_generator(0.00001, 0.3, 0.001)
    }
    clust = KMeansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedoids():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "tolerance": triangular_generator(0.00001, 0.3, 0.001)
    }
    clust = KMedoidsWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedians():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "tolerance": triangular_generator(0.00001, 0.3, 0.001)
    }
    clust = KMediansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_em():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "tolerance": triangular_generator(0.00001, 0.3, 0.00001)
    }
    clust = ExpectationMaximizationWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_bsas():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "threshold": triangular_generator(0.0, 1, 0.5)
    }
    clust = BSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_mbsas():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "threshold": triangular_generator(0.0, 1, 0.5)
    }
    clust = MBSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_ttsas():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "threshold_1": triangular_generator(0.0, 1, 0.3),
        "threshold_2": triangular_generator(0.0, 1, 0.8)
    }
    clust = TTSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_rock():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "eps": randint(0.0, 1),
        "threshold": triangular_generator(0.0, 1, 0.5)
    }
    clust = RockWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_som():
    grid_params = {
        "n_clusters": triangular_generator(4, 100, 15),
        "epoch": randint(10, 300)
    }
    clust = SOMSCWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


def cluster_hdbscan():
    grid_params = {
        "min_cluster_size": triangular_generator(5, 100, 15),
        "min_samples": randint(10, 300),
        "alpha": triangular_generator(0.1, 2.0, 1.0),
        "leaf_size": triangular_generator(5, 100, 40)
    }
    clust = HDBSCAN(metric='precomputed', memory=CACHE_PATH, core_dist_n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=ShuffleSplit(test_size=0.1, n_splits=1),
                                  n_jobs=-1, scoring=cv_silhouette_scorer, refit=True, n_iter=50)

    return searcher


# @profile
def bench_estimator(estimator, base_data, precomputed, spectral):
    if precomputed:
        data = metrics.pairwise_distances(base_data, metric='jaccard', n_jobs=-1)
        if spectral:
            data = np.exp(- data ** 2 / (2. * 1.0 ** 2))
    else:
        data = base_data

    start_time = time.time()
    estimator.fit(data)

    # convert clusters to representatives taking the intersection of all cluster points
    representatives = np.empty(shape=(0, base_data.shape[1]))
    for label_ in set(estimator.labels_):
        attrib_union = np.zeros(shape=(1, base_data.shape[1]))
        for i, elem in enumerate(estimator.labels_):
            if elem == label_:
                attrib_union = np.logical_or(attrib_union, base_data[i])
        representatives = np.vstack((representatives, attrib_union))

    hierarchy = cluster_agglomerative()
    hierarchy.fit_predict(representatives)
    time_taken = time.time() - start_time
    print(estimator.labels_)
    print(hierarchy.children_)

    return hierarchy, set(estimator.labels_)


# In: result of clustering, Out: results of TED
def compute_ted(children: np.array, n_clusters, dataset: Dataset) :
    # apted

    with open(dataset.value[1], "r") as read_file:
        groundtruth_tree = read_file.readline()

    result_tree = create_bracket_tree_from_agglo(children, n_clusters)

    command = "java -jar " + BASE + "/lib/apted.jar -t " + groundtruth_tree + " " + result_tree
    args = split(command)
    with Popen(args, stdout=PIPE) as apted:
        return apted.stdout.read().decode("utf-8")


# in: preproc. data, out: result of cv as accuracy
def cross_validate():
    raise NotImplementedError


def find_matching_brack(res_string, fix_point_idx):
    # TODO with stack
    i = fix_point_idx
    br_count = 1
    while True:
        if res_string[i] == "{":
            br_count += 1
        elif res_string[i] == "}":
            br_count -= 1

        if br_count == 0:
            return i
        else:
            i += 1


def create_bracket_tree_from_agglo(children: np.array, n_clusters):
    print(n_clusters)
    result = ""
    i = n_clusters-1
    for merge in children:
        if merge[0] > n_clusters-1 and merge[1] > n_clusters-1:

            fix_1 = result.find("{" + str(merge[0])) + 1
            matching_bracket_1 = find_matching_brack(result, fix_1)
            fix_2 = result.find("{" + str(merge[1])) + 1
            matching_bracket_2 = find_matching_brack(result, fix_2)

            temp = result
            i += 1
            # FIXME here
            print(temp)
            result = result[0: fix_1 - 1] + "{" + str(i) + result[fix_1 - 1: matching_bracket_1 + 1] + \
                     result[fix_2 - 1: matching_bracket_2 + 1] + "}"
            print(result)
            result = result + temp[matching_bracket_1 + 1: fix_2 - 1] + temp[matching_bracket_2 + 1:] if fix_1 < fix_2 \
                else result + temp[matching_bracket_2 + 1: fix_1 - 1] + temp[matching_bracket_2 + 1:]
            print(result)
        elif merge[0] > n_clusters-1 or merge[1] > n_clusters-1:
            fix_point = result.find("{" + str(merge[0])) + 1 if merge[0] > n_clusters-1 else result.find("{" + str(merge[1])) + 1
            matching_bracket = find_matching_brack(result, fix_point)
            base_clust = merge[0] if merge[0] < n_clusters else merge[1]

            i += 1
            result = result[0: fix_point - 1] + "{" + str(i) + result[fix_point - 1: matching_bracket + 1] + "{" \
                     + str(base_clust) + "}}" + result[matching_bracket + 1:]
        else:
            # Two of the base clusters are merged, declare a "new" intermediate one for the merged
            i += 1
            result += "{" + str(i) + "{" + str(merge[0]) + "}{" + str(merge[1]) + "}}"

    print(result)
    return result


def main(n_samples: int, dataset: Dataset):
    logger = logging.getLogger("clusering_survey")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('clustering_survey.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Generating/Sampling and loading data")

    data = load(n_samples, dataset)
    vectorized_data = CountVectorizer(binary=True).fit_transform(data).toarray()

    distance_matrix = metrics.pairwise_distances(vectorized_data, metric='jaccard', n_jobs=-1)

    # precomputed = False
    spectral = False

    searcher = cluster_dbscan()
    searcher.fit(distance_matrix)

    estimator = searcher.best_estimator_
    logger.info(estimator)
    params = searcher.best_params_
    logger.info(params)
    hierarchy, unique_labels = bench_estimator(estimator, vectorized_data, True, spectral)

    print(compute_ted(hierarchy.children_, len(unique_labels), dataset))

    # for searcher in [cluster_kmeans(), cluster_kmedians(), cluster_kmedoids(), cluster_bsas(), cluster_mbsas(),
    #                 cluster_ttsas(), cluster_em(), cluster_affinity_prop(), cluster_spectral(), cluster_rock(),
    #                 cluster_dbscan(), cluster_optics(), cluster_som(), cluster_agglomerative(), cluster_hdbscan()]:
    #    if isinstance(searcher.get_params()['estimator'], SpectralClustering):
    #        searcher.fit(np.exp(- distance_matrix ** 2 / (2. * 1.0 ** 2)))
    #        precomputed = True
    #        spectral = True
    #    elif 'precomputed' in searcher.get_params()['param_grid'][0]['cluster'][0].get_params().values():
    #        searcher.fit(distance_matrix)
    #        precomputed = True
    #    else:
    #        searcher.fit(vectorized_data)
    #    estimator = searcher.best_estimator_
    #    logger.info(estimator)
    #    params = searcher.best_params_
    #    logger.info(params)

    #    result, time_taken = bench_estimator(estimator, params, vectorized_data, precomputed, spectral)
    #    logger.info("Time taken to fit: " + str(time_taken))
    # compute ted & log
    # visualize & safe img
    #    precomputed = False
    #    spectral = False


if __name__ == '__main__':
    main(1024, Dataset.SYNTHETIC)


def cluster_trestle():
    # start_time = time.time()
    # time_taken = time.time() - start_time
    # concept formation
    raise NotImplementedError


def cluster_cobweb_3():
    # start_time = time.time()
    # time_taken = time.time() - start_time
    # concept formation
    raise NotImplementedError


def visualize_dendro(children: np.array, labels: np.array, distance: np.array, path: str):
    no_of_observations = np.arange(2, children.shape[0] + 2)
    # FIXME adjust distance: is it right like that
    # TODO include condensed dendro
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(linkage_matrix, labels=labels)
    plt.show()
    plt.savefig(path + "Dendrogram.svg")


# TODO Refactor and find way to visualize all kinds propperly
def visualize_clusters(path: str, labels: np.array, projected_data: np.array):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    cmap = cm.get_cmap("Spectral")
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = projected_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

    plt.title('Number of clusters: %d' % n_clusters_)
    plt.savefig(path + "Clusters.svg")
    plt.show()
    # # DBSCAN Plotting and measures
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    #
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    #
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(transformed_data, labels))
    #
    # unique_labels = set(labels)
    # cmap = cm.get_cmap("Spectral")
    # colors = [cmap(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = projected_data[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = projected_data[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    #
    # plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig(IMG_PATH + "dbscan_clusters.svg")
    # plt.show()
    # plt.title('Hierarchical Clustering Dendrogram')
    #
    # children = agglo.children_
    #
    # # Distances between each pair of children
    # # Since we don't have this information, we can use a uniform one for plotting
    # distance = np.arange(children.shape[0])
    #
    # # The number of observations contained in each cluster level
    # no_of_observations = np.arange(2, children.shape[0] + 2)
    #
    # # Create linkage matrix and then plot the dendrogram
    # linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    #
    # # Plot the corresponding dendrogram
    # dendrogram(linkage_matrix, labels=birch.labels_)
    # plt.savefig(IMG_PATH + "agglo_dendro.svg")
    #
    # plt.figure()
    # labels = birch.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # unique_labels = set(labels)
    # cmap = cm.get_cmap("Spectral")
    # colors = [cmap(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    #
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = transformed_data[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=10)
    #
    # plt.title('BIRCH + Agglo: Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig(IMG_PATH + "agglo_clust.svg")
    # plt.show()
    # labels = hdb.labels_
    #
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    #
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(transformed_data, labels))
    # print("Cluster  stability per cluster:")
    # for val in hdb.cluster_persistence_:
    #     print("\t %0.3f" % val)
    #
    # unique_labels = set(labels)
    # cmap = cm.get_cmap("Spectral")
    # colors = [cmap(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = projected_data[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = projected_data[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    # plt.title('HDBSCAN*: Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig(IMG_PATH + str(i) + "hdbscan_clusters.svg")
    # plt.show()
    # plt.figure()
    # hdb.single_linkage_tree_.plot()
    # plt.savefig(IMG_PATH + str(i) + "hdbscan_dendro.svg")
    # plt.show()
    # plt.figure()
    # hdb.condensed_tree_.plot()
    # plt.savefig(IMG_PATH + str(i) + "hdbscan_condensed.svg")
    # plt.show()


def transform_numeric(vectorized_data: np.array) -> np.array:
    transformed_data = np.array(vectorized_data)
    pca_data = PCA(n_components=0.7, svd_solver='full').fit_transform(transformed_data)
    tsne_data = TSNE(metric='jaccard').fit_transform(pca_data)

    return tsne_data
