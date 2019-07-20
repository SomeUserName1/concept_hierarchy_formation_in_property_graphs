import json
import logging
import random
import time
from enum import Enum
from os import path
import os
from shlex import split
from subprocess import Popen, PIPE
from typing import List

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import randint, uniform
from sklearn import metrics
from sklearn.cluster import DBSCAN, AffinityPropagation, SpectralClustering, OPTICS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV

from src.pyclustering_wrapper import KMeansWrapper, KMediansWrapper, KMedoidsWrapper, ExpectationMaximizationWrapper, \
    BSASWrapper, MBSASWrapper, TTSASWrapper, RockWrapper, SOMSCWrapper

# When you want to run this you need to recompile hdbscan: go to lib/hdbscan folder & run python3 setup.py install
from hdbscan import HDBSCAN, RobustSingleLinkage, condense_tree, label
from hdbscan.plots import CondensedTree
from concept_formation.trestle import TrestleTree
from concept_formation.cluster import cluster
from concept_formation.visualize import visualize as visualize_trestle
from concept_formation.visualize import visualize_clusters as visualize_trestle_clusters

# ____________ CONSTANTS: Paths ________________--
BASE = "/home/someusername/Nextcloud/workspace/uni/8/bachelor_project"
IMG_BASE = BASE + "/doc/img/"
CACHE_PATH = "/tmp/"
log = open(path.join(BASE, "doc") + 'clustering_survey_memory.log', 'w')


class Dataset(Enum):
    SYNTHETIC = (BASE + "/data/synthetic.json", BASE + "/data/synthetic.tree")
    NOISY_SYNTHETIC = (BASE + "/data/synthetic_noisy.json", BASE + "/data/synthetic.tree")
    YELP = (BASE + "/data/business.json", BASE + "/data/business.tree")


def two_step(vectorized_data, distance_matrix, dataset, n_samples, noise):
    for searcher in [cluster_affinity_prop(), cluster_spectral(n_samples), cluster_dbscan(n_samples),
                     cluster_optics(n_samples)]:
        # cluster_kmeans(n_samples), cluster_kmedians(n_samples), cluster_kmedoids(n_samples), cluster_rock(n_samples),
        #            cluster_bsas(n_samples), cluster_mbsas(n_samples), cluster_ttsas(n_samples), cluster_em(n_samples),
        #            cluster_som(n_samples)

        logger.info("======================" + type(searcher.estimator).__name__ +
                    " RobustSingleLinkage with Noise? " + str(noise) + "====================================")
        precomputed = False
        spectral = False

        if isinstance(searcher.get_params()['estimator'], SpectralClustering):
            searcher.fit(np.exp(- distance_matrix ** 2 / (2. * 1.0 ** 2)))
            precomputed = True
            spectral = True
        elif 'precomputed' in searcher.get_params().values():
            searcher.fit(distance_matrix)
            precomputed = True
        else:
            searcher.fit(vectorized_data)

        estimator = searcher.best_estimator_
        logger.info(estimator)

        visualize_clusters(estimator, vectorized_data, path.join(IMG_BASE, type(estimator).__name__, str(noise)))

        start = time.time()
        rsl = bench_two_step_estimator(estimator, vectorized_data, precomputed, spectral)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        compute_ted(
            children_or_tree=rsl.cluster_hierarchy_._linkage[:, :2].astype(int),
            n_clusters=len(set(estimator.labels_)), dataset=dataset)

        visualize(rsl, path.join(IMG_BASE + type(estimator).__name__), noise)


def single(vectorized_data, dataset, noise, n_samples):
    for searcher in [cluster_robust_single_linkage(), cluster_hdbscan(n_samples)]:
        logger.info("======================== " + type(searcher.estimator).__name__ + " with Noise? " + str(noise)
                    + " ==========================")

        searcher.fit(vectorized_data)
        estimator = searcher.best_estimator_
        logger.info(estimator)

        start = time.time()
        bench_single_estimator(estimator, vectorized_data)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        num_initial_clusters = vectorized_data.shape[0]

        if isinstance(searcher.get_params()['estimator'], RobustSingleLinkage):
            merge_mat = estimator.cluster_hierarchy_._linkage[:, :2].astype(int)

            p = path.join(IMG_BASE, type(estimator).__name__)

            compute_ted(children_or_tree=merge_mat, n_clusters=num_initial_clusters, dataset=dataset)
            visualize(estimator, p, noise)

        else:
            merge_mat = estimator.single_linkage_tree_._linkage[:, :2].astype(int)

            p = path.join(IMG_BASE, type(estimator).__name__, str(noise))
            if not os.path.exists(p):
                os.makedirs(p)

            compute_ted(children_or_tree=merge_mat, n_clusters=num_initial_clusters, dataset=dataset)

            visualize_clusters(estimator, vectorized_data, p)

            plt.figure()
            estimator.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
            plt.savefig(path.join(p, "noise_" + str(noise) + "_dendro"))
            plt.clf()

            plt.figure()
            estimator.condensed_tree_.plot()
            plt.savefig(path.join(p, "noise_" + str(noise) + "_condensed"))
            plt.clf()

            plt.figure()
            estimator.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
            plt.savefig(path.join(p, "noise_" + str(noise) + "_extracted"))
            plt.close('all')


def main(n_samples: int, dataset: Dataset, width=3, depth=2):
    for noise in [False, True]:
        logger.info("############ Noise = " + str(noise) + "#######################################################")
        logger.info("Generating/Sampling, Loading and Vectorizing Data")
        data = load(n_samples, dataset, noise, width, depth)

        vectorized_data = np.array(CountVectorizer(binary=True).fit_transform(data).toarray())
        distance_matrix = metrics.pairwise_distances(vectorized_data.astype(int), metric='l1', n_jobs=-1)

        single(vectorized_data, dataset, noise, n_samples)

        logger.info("==================== Two Step =========================")
        two_step(vectorized_data, distance_matrix, dataset, n_samples, noise)

        logger.info("================== Conceptual =======================")
        dataset = Dataset.SYNTHETIC if not noise else Dataset.NOISY_SYNTHETIC
        cluster_synthetic_trestle(dataset)


# ____________________________________Generating & Loading __________________________________________

def generate_synthetic(depth: int, width: int = 2, iteration: int = 10, m_path: str = BASE + "/data/",
                       rem_labels: int = 1, add_labels: int = 0, alter_labels: int = 1, prob: float = 0.33):
    command = "java -jar " + BASE + "/lib/synthetic_data_generator.jar -p '" + m_path + "' -d " + str(depth) + " -w " \
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
    m_path = Dataset.SYNTHETIC.value[0] if not noisy else Dataset.NOISY_SYNTHETIC.value[0]
    with open(m_path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]
    return labels


def load(n_samples: int, dataset: Dataset, noise: bool, width: int = 3, depth: int = 2) -> List[List[str]]:
    if dataset == Dataset.SYNTHETIC or dataset == Dataset.NOISY_SYNTHETIC:
        iteration = int(n_samples / (width ** depth))
        iteration = iteration if iteration > 0 else 1
        generate_synthetic(width=width, depth=depth, iteration=iteration)
        return open_synthetic(noise)
    else:
        return sample_yelp(n_samples)


# ________________________ Clustering _____________________________

# __________________________End-to-End
# ___________________________Conceptual
@profile(stream=log)
def cluster_synthetic_trestle(dataset):
    p = path.join(IMG_BASE, "trestle")
    p1 = path.join(IMG_BASE, "trestle" , "tree")

    if not os.path.exists(p):
        os.makedirs(p)

    if not os.path.exists(p1):
        os.makedirs(p1)

    with open(dataset.value[0], "r") as read_file:
        data = json.load(read_file)

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
    visualize_trestle(tree, dst=p1)
    compute_ted(children_or_tree=tree, dataset=dataset, trestle=True)

    clustering = cluster(tree, data, mod=False)
    logger.info("inferred clusters by Trestle: ")
    logger.info(clustering[0])

    visualize_trestle_clusters(tree, clustering[0], dst=p)


def cluster_robust_single_linkage():
    grid_params = {
        # "cut": uniform(),
        # "k": randint(3, 25),
        # "alpha": uniform(loc=1, scale=3),
        # "gamma": randint(3, 20),
    }
    clust = RobustSingleLinkage(core_dist_n_jobs=1, metric='l1')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1,
                                  scoring=cv_scorer, refit=True, n_iter=1)

    return searcher


def cluster_hdbscan(n_samples):
    grid_params = {
        "min_cluster_size": randint(2, 5),
        "min_samples": randint(2, 5),
        # "alpha": uniform(loc=0.1, scale=5),
        # "leaf_size": randint(5, round(0.3 * n_samples) + 5)
    }
    clust = HDBSCAN(metric='l1', memory=CACHE_PATH, core_dist_n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


# ___________________________ Pre-Clustering

def cluster_affinity_prop():
    grid_params = {
        "damping": uniform(loc=0.5, scale=0.5)
    }
    clust = AffinityPropagation(affinity='precomputed')
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)
    return searcher


def cluster_spectral(n_samples):
    grid_params = {
        "n_clusters": randint(5, n_samples / 2 + 5),
    }
    clust = SpectralClustering(affinity='precomputed', eigen_solver='arpack', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)
    return searcher


def cluster_dbscan(n_samples):
    grid_params = {
        "eps": uniform(loc=0.1, scale=5.0),
        "min_samples": randint(2, round(0.1 * n_samples) + 2),
        "leaf_size": randint(5, round(0.1 * n_samples) + 5)
    }
    clust = DBSCAN(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(),
                                  n_jobs=-1, scoring=cv_scorer, error_score='raise', refit=True, n_iter=100)
    return searcher


def cluster_optics(n_samples):
    grid_params = {
        "min_samples": randint(2, round(0.1 * n_samples) + 2),
        "leaf_size": randint(5, 100),
        "min_cluster_size": uniform(0, 0.1)
    }
    clust = OPTICS(metric='precomputed', n_jobs=-1)
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=10)

    return searcher


def cluster_kmeans(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMeansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedoids(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMedoidsWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_kmedians(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = KMediansWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_em(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "tolerance": uniform(0.00001, 0.1)
    }
    clust = ExpectationMaximizationWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_bsas(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "threshold": uniform(0, 1)
    }
    clust = BSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_mbsas(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "threshold": uniform(0, 1)
    }
    clust = MBSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_ttsas(n_samples):
    grid_params = {
        "max_n_clusters": randint(4, (n_samples / 2) + 4),
        "threshold": uniform(loc=0, scale=0.5),
        "threshold_2": uniform(0.5, 1)
    }
    clust = TTSASWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_rock(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "eps": uniform(0.0, 1),
        "threshold": uniform(0.0, 1)
    }
    clust = RockWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
                                  n_jobs=-1, scoring=cv_scorer, refit=True, n_iter=50)

    return searcher


def cluster_som(n_samples):
    grid_params = {
        "n_clusters": randint(4, (n_samples / 2) + 4),
        "epoch": randint(10, 300)
    }
    clust = SOMSCWrapper()
    searcher = RandomizedSearchCV(clust, param_distributions=grid_params, cv=DisabledCV(), error_score='raise',
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
        data = metrics.pairwise_distances(base_data.astype(bool), metric='l1', n_jobs=-1)
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

    logger.info("First step clustering labels: ")
    logger.info(estimator.labels_)
    logger.info("Representative matrix: ")
    logger.info(representatives)

    hierarchy_searcher = cluster_robust_single_linkage()
    hierarchy_searcher.fit(representatives)
    rsl = hierarchy_searcher.estimator

    rsl.fit(representatives)

    return rsl


# In: result of clustering, Out: results of TED
def compute_ted(children_or_tree, dataset: Dataset, trestle: bool = False, n_clusters=0):
    with open(dataset.value[1], "r") as read_file:
        ground_truth = read_file.read()

    if trestle:
        result = create_bracket_tree_trestle(children_or_tree)
    else:
        result = create_bracket_tree_rsl(children_or_tree, n_clusters)

    command = "java -jar " + BASE + "/lib/apted.jar -t " + ground_truth + " " + result
    args = split(command)
    with Popen(args, stdout=PIPE) as apted:
        logger.info("Tree Edit Distance: " + apted.stdout.read().decode("utf-8"))


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


def find_fix(result, elem):
    fix = result.find("{" + str(elem)) + 1

    while result[fix + len(str(elem))].isdigit():
        fix = result.find("{" + str(elem)) + 1

    return fix


def create_bracket_tree_rsl(children: np.array, n_clusters):
    logger.info(n_clusters)
    logger.info(children)
    result = ""
    i = n_clusters - 1
    for merge in children:
        if merge[0] > n_clusters - 1 and merge[1] > n_clusters - 1:

            fix_1 = find_fix(result, merge[0])
            matching_bracket_1 = find_matching_brack(result, fix_1)
            fix_2 = find_fix(result, merge[1])
            matching_bracket_2 = find_matching_brack(result, fix_2)

            temp = result
            i += 1

            if fix_1 < fix_2:
                moved = result[0: fix_1 - 1] + "{" + str(i) + result[fix_1 - 1: matching_bracket_1 + 1] \
                        + result[fix_2 - 1: matching_bracket_2 + 1] + "}"
                result = moved + temp[matching_bracket_1 + 1: fix_2 - 1] + temp[matching_bracket_2 + 1:]

            else:
                moved = result[0: fix_2 - 1] + "{" + str(i) + result[fix_2 - 1: matching_bracket_2 + 1] \
                        + result[fix_1 - 1: matching_bracket_1 + 1] + "}"
                result = moved + temp[matching_bracket_2 + 1: fix_1 - 1] + temp[matching_bracket_1 + 1:]

        elif merge[0] > n_clusters - 1 or merge[1] > n_clusters - 1:
            fix_point = find_fix(result, merge[0]) if merge[0] > n_clusters - 1 else find_fix(result, merge[1])
            matching_bracket = find_matching_brack(result, fix_point)
            base_clust = merge[0] if merge[0] < n_clusters else merge[1]

            i += 1
            result = result[0: fix_point - 1] + "{" + str(i) + result[fix_point - 1: matching_bracket + 1] + "{" + \
                     str(base_clust) + "}}" + result[matching_bracket + 1:]
        else:
            # Two of the base clusters are merged, declare a "new" intermediate one for the merged
            i += 1
            result += "{" + str(i) + "{" + str(merge[0]) + "}{" + str(merge[1]) + "}}"

    logging.info("Cluster result based bracket_tree ")
    logger.info(result)
    return result


def create_bracket_tree_trestle(tree):
    result = ""
    i = 0
    prev_tab_count = -1
    tab_stack = 0

    for elem in tree.__str__().split('\n'):
        cur_tab_count = elem.count('\t')
        if prev_tab_count == cur_tab_count:
            result += "}"
        elif prev_tab_count < cur_tab_count:
            tab_stack += 1
        else:
            delta = prev_tab_count - cur_tab_count
            for _ in range(0, delta + 1):
                result += "}"
            tab_stack -= delta

        result += elem[:elem.find("{")].replace("|-", "{" + str(i)).replace("\t", "")

        prev_tab_count = cur_tab_count

        i += 1
    logger.info("trestle result based bracket tree:")
    logger.info(result)
    return result


def visualize_clusters(estimator, data, p_path):
    labels = estimator.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    unique_labels = set(labels)
    cmap = cm.get_cmap("Spectral")
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
    print("Calinski Harabasz Score: %0.3f" % metrics.calinski_harabasz_score(data, labels))

    projected_data = transform_numeric(data)

    plt.figure()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = projected_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

    plt.title(type(estimator).__name__ + ' Number of clusters: %d' % n_clusters_)
    plt.savefig(p_path + "_clusters.svg")
    plt.close('all')


def visualize(estimator, m_path, noise):
    if not path.exists(m_path):
        os.makedirs(m_path)
    p_path = path.join(m_path, "noise_" + str(noise))

    plt.figure()
    dendrogram(estimator.cluster_hierarchy_._linkage)
    plt.savefig(p_path + "_scipy_dendro.svg")

    plt.figure()
    estimator.cluster_hierarchy_.plot()
    plt.savefig(p_path + "_rsl_dendro.svg")
    plt.clf()

    plt.figure()
    ct = CondensedTree(condense_tree(estimator.cluster_hierarchy_._linkage, 2))
    ct.plot()
    plt.savefig(p_path + "_condensed.svg")
    plt.clf()

    plt.figure()
    ct.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.savefig(p_path + "_extracted.svg")
    plt.close('all')


def transform_numeric(vectorized_data: np.array) -> np.array:
    tsne_data = TSNE(metric='l1').fit_transform(vectorized_data)

    return tsne_data


if __name__ == '__main__':
    logger = logging.getLogger("clusering_survey")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(path.join(BASE, "doc", "clusering_survey.log"), mode='w')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    main(n_samples=(3 ** 4), dataset=Dataset.SYNTHETIC, width=3, depth=4)
