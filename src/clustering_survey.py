import json
import logging
import random
import time
from enum import Enum
from shlex import split
from subprocess import Popen, PIPE
from typing import List

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
from scipy.stats import randint, uniform
from sklearn import metrics
from sklearn.cluster import DBSCAN, AffinityPropagation, SpectralClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV

from src.pyclustering_wrapper import KMeansWrapper, KMediansWrapper, KMedoidsWrapper, ExpectationMaximizationWrapper, \
    BSASWrapper, MBSASWrapper, TTSASWrapper, RockWrapper, SOMSCWrapper

# When you want to run this you need to recompile hdbscan: go to lib/hdbscan folder & run python3 setup.py install
from hdbscan import HDBSCAN, RobustSingleLinkage, condense_tree, label
from concept_formation.trestle import TrestleTree
from concept_formation.cluster import cluster

# ____________ CONSTANTS: Paths ________________--
BASE = "/home/someusername/Nextcloud/workspace/uni/8/bachelor_project"
IMG_BASE = BASE + "/doc/img/"
CACHE_PATH = "/tmp/"
log = open('clustering_survey.log', 'w+')


class Dataset(Enum):
    SYNTHETIC = (BASE + "/data/synthetic.json", BASE + "/data/synthetic.tree")
    NOISY_SYNTHETIC = (BASE + "/data/synthetic_noisy.json", BASE + "/data/synthetic.tree")
    YELP = (BASE + "/data/business.json", BASE + "/data/business.tree")


def two_step(vectorized_data, distance_matrix, dataset):
    for searcher in [cluster_kmeans(), cluster_kmedians(), cluster_kmedoids(), cluster_bsas(), cluster_mbsas(),
                     cluster_ttsas(), cluster_em(), cluster_affinity_prop(), cluster_spectral(), cluster_rock(),
                     cluster_dbscan(), cluster_optics(), cluster_som()]:
        logger.info("====================== " + searcher + " RobustSingleLinkage ====================================")
        precomputed = False
        spectral = False

        if isinstance(searcher.get_params()['estimator'], SpectralClustering):
            searcher.fit(np.exp(- distance_matrix ** 2 / (2. * 1.0 ** 2)))
            precomputed = True
            spectral = True
        elif 'precomputed' in searcher.get_params()['param_grid'][0]['cluster'][0].get_params().values():
            searcher.fit(distance_matrix)
            precomputed = True
        else:
            searcher.fit(vectorized_data)

        estimator = searcher.best_estimator_
        logger.info("Found parameters: " + estimator)

        start = time.time()
        rsl = bench_two_step_estimator(estimator, vectorized_data, precomputed, spectral)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        logger.info("Tree Edit Distance: " + compute_ted(rsl.cluster_hierarchy_._linkage[:, :2].astype(int),
                                                         len(estimator.labels_), dataset))
        vis_rsl(estimator, IMG_BASE + type(estimator).__name__)


def single(vectorized_data, dataset):
    for searcher in [cluster_robust_single_linkage(), cluster_hdbscan()]:
        logger.info("======================== " + searcher + " ==========================")

        searcher.fit(vectorized_data)
        estimator = searcher.best_estimator_
        logger.info(estimator)

        start = time.time()
        bench_single_estimator(estimator, vectorized_data)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        logger.info("Tree Edit distance: " + compute_ted(estimator.cluster_hierarchy_._linkage[:, :2].astype(int),
                                                         vectorized_data.shape[0], dataset))
        if isinstance(searcher.get_params()['estimator'], RobustSingleLinkage):
            vis_rsl(estimator, IMG_BASE + type(estimator).__name__)

        else:
            plt.figure()
            estimator.single_linkage_tree_plot(cmap='viridis', colorbar=True)
            plt.savefig(IMG_BASE + type(estimator).__name__ + "_dendro")
            plt.clf()

            plt.figure()
            estimator.condensed_tree_.plot()
            plt.savefig(IMG_BASE + type(estimator).__name__ + "_condensed")
            plt.clf()

            plt.figure()
            estimator.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
            plt.savefig(IMG_BASE + type(estimator).__name__ + "_extracted")
            plt.clf()


def main(n_samples: int, dataset: Dataset):
    for noise in [True, False]:
        logger.info("############ Noise = " + str(noise) + "#######################################################")
        logger.info("Generating/Sampling, Loading and Vectorizing Data")
        data = load(n_samples, dataset, noise)

        vectorized_data = CountVectorizer(binary=True).fit_transform(data).toarray().astype(bool)
        distance_matrix = metrics.pairwise_distances(vectorized_data, metric='jaccard', n_jobs=-1)

        single(vectorized_data, data)

        logger.info("==================== Two Step =========================")
        two_step(vectorized_data, distance_matrix, dataset)

        logger.info("================== Conceptual =======================")
        path = Dataset.SYNTHETIC.value[0] if not noise else Dataset.NOISY_SYNTHETIC.value[0]
        with open(path, "r") as read_file:
            data = json.load(read_file)
        cluster_synthetic_trestle(data)


# ____________________________________Generating & Loading __________________________________________

def generate_synthetic(depth: int, width: int = 2, iteration: int = 10, path: str = BASE + "/data/",
                       rem_labels: int = 1, add_labels: int = 0, alter_labels: int = 1, prob: float = 0.33):
    command = "java -jar " + BASE + "/lib/synthetic_data_generator.jar -p '" + path + "' -d " + str(depth) + " -w " \
              + str(width) + " -i " + str(iteration) + " -n " + str(rem_labels) + " " + str(add_labels) + " " + \
              str(alter_labels) + " -pr " + str(prob)
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
    path = Dataset.SYNTHETIC.value[0] if not noisy else Dataset.NOISY_SYNTHETIC.value[0]
    with open(path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]
    return labels


def load(n_samples: int, dataset: Dataset, noise: bool) -> List[List[str]]:
    if dataset == Dataset.SYNTHETIC or dataset == Dataset.NOISY_SYNTHETIC:
        generate_synthetic(width=3, depth=2, iteration=100)  # int(math.log2(n_samples)))
        return open_synthetic(noise)
    else:
        return sample_yelp(n_samples)


# ________________________ Clustering _____________________________

# __________________________End-to-End
# ___________________________Conceptual
@profile(stream=log)
def cluster_synthetic_trestle(data):
    for entry in data:
        entry["id"] = "_" + str(entry["id"])
        del entry["id"]
        for i, label in enumerate(entry["labels"].split()):
            entry["label_" + str(i)] = label

        del entry["labels"]

    tree = TrestleTree()

    start = time.time()
    tree.fit(data)
    total = time.time() - start

    logger.info("Took " + str(total) + " s! Trestle tree: ")
    logger.info(tree)

    clustering = cluster(tree, data, mod=False)
    logger.info("infered clusters by Trestle: " + clustering)


def cluster_robust_single_linkage():
    grid_params = {
        "cut": uniform(),
        "k": randint(3, 25),
        "alpha": uniform(loc=1, scale=3),
        "gamma": randint(3, 20),
    }
    clust = RobustSingleLinkage(metric='jaccard', core_dist_n_jobs=1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1,
                                  scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_hdbscan():
    grid_params = {
        "min_cluster_size": randint(5, 30),
        "min_samples": randint(10, 300),
        "alpha": uniform(loc=0.1, scale=2),
        "leaf_size": randint(5, 50)
    }
    clust = HDBSCAN(metric='precomputed', memory=CACHE_PATH, core_dist_n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


# ___________________________ Pre-Clustering

def cluster_affinity_prop():
    grid_params = {
        "damping": uniform(loc=0.5, scale=0.5)
    }
    clust = AffinityPropagation(affinity='precomputed')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)
    return searcher


def cluster_spectral():
    grid_params = {
        "n_clusters": randint(5, 30, 15),
    }
    clust = SpectralClustering(affinity='precomputed', eigen_solver='amg', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)
    return searcher


def cluster_dbscan():
    grid_params = {
        "eps": uniform(),
        "min_samples": randint(2, 8),
        "leaf_size": randint(3, 10)
    }
    clust = DBSCAN(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, error_score='raise', refit=True, n_iter=100)
    return searcher


def cluster_optics():
    grid_params = {
        "max_eps": randint(0.1, 1),
        "min_samples": randint(0.01, 1),
        "leaf_size": randint(5, 100),
        "xi": uniform(loc=0, scale=0.2),
        "min_cluster_size": uniform(0, 0.3)
    }
    clust = OPTICS(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmeans():
    grid_params = {
        "n_clusters": randint(4, 30),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMeansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedoids():
    grid_params = {
        "n_clusters": randint(4, 30),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMedoidsWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedians():
    grid_params = {
        "n_clusters": randint(4, 30),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMediansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_em():
    grid_params = {
        "n_clusters": randint(4, 30),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = ExpectationMaximizationWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_bsas():
    grid_params = {
        "n_clusters": randint(4, 30),
        "threshold": uniform(0, 1)
    }
    clust = BSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_mbsas():
    grid_params = {
        "n_clusters": randint(4, 30),
        "threshold": uniform(0, 1)
    }
    clust = MBSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_ttsas():
    grid_params = {
        "n_clusters": randint(4, 30),
        "threshold": uniform(loc=0, scale=0.5),
        "threshold_2": uniform(0.5, 1)
    }
    clust = TTSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_rock():
    grid_params = {
        "n_clusters": randint(4, 30),
        "eps": uniform(0.0, 1),
        "threshold": uniform(0.0, 1)
    }
    clust = RockWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_som():
    grid_params = {
        "n_clusters": randint(4, 30),
        "epoch": randint(10, 300)
    }
    clust = SOMSCWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

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


# _____________________ Benchmarking __________________________________--
@profile(stream=log)
def bench_single_estimator(estimator, vectorized_data):
    estimator.fit(vectorized_data)


@profile(stream=log)
def bench_two_step_estimator(estimator, base_data, precomputed, spectral):
    if precomputed:
        data = metrics.pairwise_distances(base_data, metric='jaccard', n_jobs=-1)
        if spectral:
            data = np.exp(- data ** 2 / (2. * 1.0 ** 2))
    else:
        data = base_data

    estimator.fit(data)

    # convert clusters to representatives taking the intersection of all cluster points
    representatives = np.empty(shape=(0, base_data.shape[1]))
    for label_ in set(estimator.labels_):
        attrib_union = np.ones(shape=(1, base_data.shape[1]))
        for i, elem in enumerate(estimator.labels_):
            if elem == label_:
                attrib_union = np.logical_and(attrib_union, base_data[i])
        representatives = np.vstack((representatives, attrib_union))

    logger.info("First step clustering labels: " + estimator.labels_)
    logger.info("Representative matrix: " + representatives)

    hierarchy_searcher = cluster_robust_single_linkage()
    hierarchy_searcher.fit(representatives)
    rsl = hierarchy_searcher.estimator

    rsl.fit(representatives)

    return rsl


# In: result of clustering, Out: results of TED
def compute_ted(children: np.array, n_clusters, dataset: Dataset):
    with open(dataset.value[1], "r") as read_file:
        groundtruth_tree = read_file.readline()

    result_tree = create_bracket_tree_rsl(children, n_clusters)

    command = "java -jar " + BASE + "/lib/apted.jar -t " + groundtruth_tree + " " + result_tree
    args = split(command)
    with Popen(args, stdout=PIPE) as apted:
        return apted.stdout.read().decode("utf-8")


# in: preproc. data, out: result of cv as accuracy
def cross_validate():
    raise NotImplementedError


def find_matching_brack(res_string, fix_point_idx):
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


def create_bracket_tree_rsl(children: np.array, n_clusters):
    print(children)
    result = ""
    i = n_clusters - 1
    for merge in children:
        if merge[0] > n_clusters - 1 and merge[1] > n_clusters - 1:

            fix_1 = result.find("{" + str(merge[0])) + 1
            matching_bracket_1 = find_matching_brack(result, fix_1)
            fix_2 = result.find("{" + str(merge[1])) + 1
            matching_bracket_2 = find_matching_brack(result, fix_2)

            temp = result
            i += 1

            result = result[0: fix_1 - 1] + "{" + str(i) + result[fix_1 - 1: matching_bracket_1 + 1] \
                     + result[fix_2 - 1: matching_bracket_2 + 1] + "}"

            result = result + temp[matching_bracket_1 + 1: fix_2 - 1] + temp[matching_bracket_2 + 1:] if fix_1 < fix_2 \
                else result + temp[matching_bracket_2 + 1: fix_1 - 1] + temp[matching_bracket_2 + 1:]

        elif merge[0] > n_clusters - 1 or merge[1] > n_clusters - 1:
            fix_point = result.find("{" + str(merge[0])) + 1 if merge[0] > n_clusters - 1 else result.find(
                "{" + str(merge[1])) + 1
            matching_bracket = find_matching_brack(result, fix_point)
            base_clust = merge[0] if merge[0] < n_clusters else merge[1]

            i += 1
            result = result[0: fix_point - 1] + "{" + str(i) + result[fix_point - 1: matching_bracket + 1] + "{" \
                     + str(base_clust) + "}}" + result[matching_bracket + 1:]
        else:
            # Two of the base clusters are merged, declare a "new" intermediate one for the merged
            i += 1
            result += "{" + str(i) + "{" + str(merge[0]) + "}{" + str(merge[1]) + "}}"

    print("bracket_tree " + result)
    return result


# __________________ Visualization of first step (as second is impl. by hdbscan package)
# TODO Refactor and find way to visualize all kinds properly
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


def vis_rsl(estimator, path):
    plt.figure()
    estimator.cluster_hierarchy_.plot()
    plt.savefig(path + "_dendro")
    plt.clf()

    plt.figure()
    ct = condense_tree(estimator.cluster_hierarchy_, 10)
    ct.plot()
    plt.savefig(path + "_condensed")
    plt.clf()

    plt.figure()
    ct.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.savefig(path + "_extracted")
    plt.clf()


def transform_numeric(vectorized_data: np.array) -> np.array:
    transformed_data = np.array(vectorized_data)
    pca_data = PCA(n_components=0.7, svd_solver='full').fit_transform(transformed_data)
    tsne_data = TSNE(metric='jaccard').fit_transform(pca_data)

    return tsne_data


if __name__ == '__main__':
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

    main(8, Dataset.SYNTHETIC)
