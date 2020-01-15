from __future__ import absolute_import

import time
from os import path, makedirs
import numpy as np

from concept_formation.trestle import TrestleTree
from concept_formation.visualize import visualize as visualize_trestle
# When you want to run this you need to recompile hdbscan: go to lib/hdbscan folder & run python3 setup.py install
from scipy.cluster.hierarchy import single
from scipy.spatial.distance import pdist, jaccard
from sklearn.feature_extraction.text import CountVectorizer

from algorithm_search_wrapper import *
from data_loader import load, preprocess_trestle_yelp, preprocess_trestle_synthetic
from constants import IMG_BASE, Dataset, logger, result_summary, BASE
from tree_edit_distance import compute_ted
from visualization import visualize, visualize_clusters, parse_results


# cluster_kmedoids(n_samples), cluster_spectral(n_samples), cluster_affinity_prop(), cluster_birch(),
# cluster_rock(n_samples), cluster_kmedians(n_samples),  cluster_bsas(n_samples), cluster_mbsas(n_samples),
# , cluster_som(n_samples)
def two_step(vectorized_data, dataset, n_samples, noise):
    for searcher in [cluster_optics(n_samples), cluster_kmeans(n_samples), cluster_dbscan(n_samples),
                     cluster_rsl(n_samples), cluster_hdbscan(n_samples), cluster_ttsas()]:
        logger.info("======================" + type(searcher.estimator).__name__ +
                    " SingleLinkage with Noise? " + str(noise) + "====================================")

        vectorized_data = np.array(vectorized_data).astype(bool)
        searcher.fit(vectorized_data)

        estimator = searcher.best_estimator_
        logger.info(estimator)

        visualize_clusters(estimator, vectorized_data, path.join(IMG_BASE, type(estimator).__name__), noise, dataset)

        start = time.time()
        linkage = bench_two_step_estimator(estimator, vectorized_data)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        if dataset is not dataset.YELP:
            compute_ted(
                children_or_tree=linkage,
                name=type(searcher.estimator).__name__, noise=noise, seconds=str(total), samples=n_samples,
                n_clusters=len(set(estimator.labels_)), dataset=dataset)

        visualize(linkage, path.join(IMG_BASE + type(estimator).__name__), noise, dataset)


def cluster_trestle(dataset, noise, n_samples):
    p = path.join(IMG_BASE, "trestle")
    p1 = path.join(IMG_BASE, "trestle", "tree")

    if not path.exists(p):
        makedirs(p)

    if not path.exists(p1):
        makedirs(p1)

    if dataset == Dataset.SYNTHETIC:
        dataset = Dataset.SYNTHETIC if not noise else Dataset.NOISY_SYNTHETIC
        data = preprocess_trestle_synthetic(dataset)
    else:
        data = preprocess_trestle_yelp(n_samples)

    tree = TrestleTree()

    start = time.time()
    tree.fit(data)
    total = time.time() - start

    logger.info("Took " + str(total) + " s! Trestle tree: ")
    logger.info(tree)
    visualize_trestle(tree, dst=p1)
    if dataset is not dataset.YELP:
        compute_ted(children_or_tree=tree, name="Trestle", noise=noise, seconds=str(total), samples=n_samples,
                    dataset=dataset, trestle=True)


def cluster_basic_single_linkage(diatance_matrix, dataset, noise, n_samples):
    logger.info("======= Simple Single Linkage ========")
    start = time.time()
    linkage = single(pdist(diatance_matrix, metric=jaccard))
    total = time.time() - start
    logger.info("Fitting took: " + str(total) + " s!")

    if dataset is not dataset.YELP:
        compute_ted(
            children_or_tree=linkage,
            name="SingleLinkage",
            noise=noise, seconds=str(total), samples=n_samples,
            n_clusters=diatance_matrix.shape[0], dataset=dataset)

    visualize(linkage, path.join(IMG_BASE + "Single linkage"), noise, dataset)


def start_clustering(dataset, n_samples, noise, width, depth):
    logger.info("############ Noise = " + str(noise) + "#######################################################")
    logger.info("Generating/Sampling, Loading and Vectorizing Data")
    data = load(n_samples, dataset, noise, width, depth)

    vectorized_data = np.array(CountVectorizer(binary=True).fit_transform(data).toarray()).astype(bool)

    cluster_basic_single_linkage(vectorized_data, dataset, noise, n_samples)

    logger.info("==================== Two Step =========================")
    two_step(vectorized_data, dataset, n_samples, noise)

    logger.info("================== Conceptual =======================")
    cluster_trestle(dataset, noise, n_samples)


def main():
    for dataset in [Dataset.SYNTHETIC, Dataset.YELP]:
        for width, depth in [[3, 5], [8, 3], [4, 5], [11, 3], [12, 3], [13, 3],  [7, 4], [5, 5], [4, 6], [9, 4]]:
            n_samples = width ** depth
            logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + str(n_samples) +
                        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if dataset == Dataset.SYNTHETIC:
                for noise in [0, 0.05, 0.10, 0.20, 0.33]:
                    start_clustering(dataset, n_samples, noise, width, depth)
            else:
                start_clustering(dataset, n_samples, 0, width, depth)


# _____________________ Benchmarking ____________________________________
def bench_two_step_estimator(estimator, data):
    estimator.fit(data)

    # convert clusters to representatives taking the intersection of all cluster points
    representatives = np.empty(shape=(0, data.shape[1]))
    for label_ in set(estimator.labels_):
        attrib_union = np.ones(shape=(1, data.shape[1]))
        for i, elem in enumerate(estimator.labels_):
            if elem == label_:
                attrib_union = np.logical_and(attrib_union, data[i])
        representatives = np.vstack((representatives, attrib_union))

    repr_dist = pdist(representatives, metric=jaccard)
    linkage = single(repr_dist)

    return linkage


if __name__ == '__main__':
    # main()
    parse_results("/home/someusername/Nextcloud/workspace/uni/bachelor/klopfer-bachelor/doc/clustering_survey_results.log")
    result_summary.close()
