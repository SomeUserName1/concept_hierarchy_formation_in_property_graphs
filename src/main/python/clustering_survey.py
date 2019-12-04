from __future__ import absolute_import

import time
from os import path, makedirs

from memory_profiler import profile
import matplotlib.pyplot as plt
import seaborn as sns
from concept_formation.cluster import cluster
from concept_formation.trestle import TrestleTree
from concept_formation.visualize import visualize as visualize_trestle
from concept_formation.visualize import visualize_clusters as visualize_trestle_clusters
# When you want to run this you need to recompile hdbscan: go to lib/hdbscan folder & run python3 setup.py install
from hdbscan import HDBSCAN, RobustSingleLinkage
from scipy.cluster.hierarchy import single
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import CountVectorizer

from algorithm_search_wrapper import *
from data_loader import load, preprocess_trestle_yelp, preprocess_trestle_synthetic
from src_project.main.python import IMG_BASE, Dataset, logger, CACHE_PATH
from tree_edit_distance import compute_ted
from visualization import visualize, visualize_clusters


def two_step(vectorized_data, distance_matrix, dataset, n_samples, noise):
    for searcher in [cluster_dbscan(n_samples), cluster_affinity_prop(), cluster_optics(n_samples),
                     cluster_birch(), cluster_spectral(n_samples), cluster_kmeans(n_samples),
                     cluster_kmedians(n_samples), cluster_kmedoids(n_samples), cluster_rock(n_samples),
                     cluster_bsas(n_samples), cluster_mbsas(n_samples), cluster_ttsas(), cluster_som(n_samples)]:

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

        visualize_clusters(estimator, vectorized_data, path.join(IMG_BASE, type(estimator).__name__), noise)

        start = time.time()
        rsl, linkage = bench_two_step_estimator(estimator, vectorized_data, precomputed, spectral, True)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        compute_ted(
            children_or_tree=linkage,
            name=type(searcher.estimator).__name__, noise=noise, seconds=str(total), samples=n_samples,
            n_clusters=len(set(estimator.labels_)), dataset=dataset)

        visualize(rsl, linkage, path.join(IMG_BASE + type(estimator).__name__), noise)


def one_step(vectorized_data, dataset, noise, n_samples):
    rsl = RobustSingleLinkage(metric='jaccard')
    hdb = HDBSCAN(metric='jaccard', memory=CACHE_PATH, core_dist_n_jobs=-1, min_samples=2, min_cluster_size=2)

    for algo in [rsl, hdb]:
        logger.info("======================== " + type(algo).__name__ + " with Noise? " + str(noise)
                    + " ==========================")
        # TODO: only use selected clusters by hdbscan,
        # TODO check robust single linkage
        algo.fit(vectorized_data)

        start = time.time()
        bench_single_estimator(algo, vectorized_data)
        total = time.time() - start
        logger.info("Fitting took: " + str(total) + " s")

        num_initial_clusters = vectorized_data.shape[0]

        if isinstance(algo, RobustSingleLinkage):
            linkage = algo.cluster_hierarchy_._linkage

            p = path.join(IMG_BASE, type(algo).__name__)

            compute_ted(
                children_or_tree=linkage,
                name=type(algo).__name__, noise=noise, seconds=str(total), samples=n_samples,
                n_clusters=num_initial_clusters, dataset=dataset)
            visualize(algo, linkage, p, noise)

        else:
            p = path.join(IMG_BASE, type(algo).__name__)
            if not path.exists(p):
                makedirs(p)

            compute_ted(children_or_tree=algo.single_linkage_tree_._linkage,
                        name=type(algo).__name__, noise=noise, seconds=str(total), samples=n_samples,
                        n_clusters=num_initial_clusters, dataset=dataset)

            visualize_clusters(algo, vectorized_data, p, noise)

            plt.figure()
            algo.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
            plt.savefig(path.join(p, "noise_" + str(noise) + "_dendro.pdf"))
            plt.clf()

            plt.figure()
            algo.condensed_tree_.plot()
            plt.savefig(path.join(p, "noise_" + str(noise) + "_condensed.pdf"))
            plt.clf()

            plt.figure()
            algo.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
            plt.savefig(path.join(p, "noise_" + str(noise) + "_extracted.pdf"))
            plt.close('all')


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
    compute_ted(children_or_tree=tree, name="Trestle", noise=noise, seconds=str(total), samples=n_samples,
                dataset=dataset, trestle=True)

    clustering = cluster(tree, data, mod=False)
    logger.info("inferred clusters by Trestle: ")
    logger.info(clustering[0])

    visualize_trestle_clusters(tree, clustering[0], dst=p)


def cluster_basic_single_linkage(vectorized_data, dataset, noise, n_samples):
    logger.info("======= Simple Single Linkage ========")
    rsl = None
    start = time.time()
    linkage = single(pdist(vectorized_data, metric='jaccard'))
    total = time.time() - start
    logger.info("Fitting took: " + str(total) + " s!")

    compute_ted(
        children_or_tree=linkage,
        name="SingleLinkage",
        noise=noise, seconds=str(total), samples=n_samples,
        n_clusters=vectorized_data.shape[0], dataset=dataset)

    visualize(rsl, linkage, path.join(IMG_BASE + "Single linkage"), noise)


def start_clustering(dataset, n_samples, noise, width, depth):
    logger.info("############ Noise = " + str(noise) + "#######################################################")
    logger.info("Generating/Sampling, Loading and Vectorizing Data")
    data = load(n_samples, dataset, noise, width, depth)

    vectorized_data = np.array(CountVectorizer(binary=True).fit_transform(data).toarray()).astype(bool)
    distance_matrix = metrics.pairwise_distances(vectorized_data, metric='jaccard', n_jobs=-1)

    cluster_basic_single_linkage(vectorized_data, dataset, noise, n_samples)

    one_step(vectorized_data, dataset, noise, n_samples)

    logger.info("==================== Two Step =========================")
    two_step(vectorized_data, distance_matrix, dataset, n_samples, noise)

    logger.info("================== Conceptual =======================")
    cluster_trestle(dataset, noise, n_samples)


def main():
    for dataset in [Dataset.SYNTHETIC, Dataset.YELP]:
        for width, depth in [[3, 3], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7]]:
            n_samples = width ** depth
            if dataset == Dataset.SYNTHETIC:
                for noise in [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1]:
                    start_clustering(dataset, n_samples, noise, width, depth)
            else:
                start_clustering(dataset, n_samples, 0, width, depth)


# _____________________ Benchmarking ____________________________________
def bench_single_estimator(estimator, vectorized_data):
    estimator.fit(vectorized_data)


def bench_two_step_estimator(estimator, base_data, precomputed, spectral, simple_linkage: bool):
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

    if simple_linkage:
        repr_dist = pdist(representatives, metric='jaccard')
        linkage = single(repr_dist)
        rsl = None
    else:
        rsl = RobustSingleLinkage(metric="jaccard")
        rsl.fit(representatives)
        linkage = rsl.cluster_hierarchy_._linkage

    return rsl, linkage


if __name__ == '__main__':
    main()
