from os import path, makedirs
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from hdbscan import condense_tree
from hdbscan.plots import CondensedTree

from constants import logger, result_summary, Dataset, IMG_BASE, BASE


def visualize_clusters(estimator, data, p_path, noise, dataset):
    if not path.exists(p_path):
        makedirs(p_path)
    labels = estimator.labels_

    dataset_str = "synth_" if dataset is not Dataset.YELP else "yelp_"

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    logger.info('Estimated number of clusters: %d' % n_clusters_)
    logger.info('Estimated number of noise points: %d' % n_noise_)

    unique_labels = set(labels)
    cmap = cm.get_cmap("Spectral")
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]

    logger.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
    logger.info("Calinski Harabasz Score: %0.3f" % metrics.calinski_harabasz_score(data, labels))

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
    plt.savefig(path.join(p_path, dataset_str + "noise_" + str(noise) + "_clusters.pdf"))
    plt.close('all')


def visualize(linkage, m_path, noise, dataset):
    if not path.exists(m_path):
        makedirs(m_path)

    dataset_str = "synth_" if dataset is not Dataset.YELP else "yelp_"
    p_path = path.join(m_path, dataset_str + "noise_" + str(noise))

    plt.figure()
    children = linkage[:, :2]
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0] + 2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    dendrogram(linkage_matrix)
    plt.savefig(p_path + "_equi_dendro.pdf")
    plt.clf()

    plt.figure()
    dendrogram(linkage)
    plt.savefig(p_path + "_dendro.pdf")
    plt.clf()

    plt.figure()
    ct = CondensedTree(condense_tree(linkage, 2), cluster_selection_method='eom',
                       allow_single_cluster=True)
    ct.plot()
    plt.savefig(p_path + "_condensed.pdf")
    plt.clf()

    plt.figure()
    ct.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.savefig(p_path + "_extracted.pdf")
    plt.close('all')


def transform_numeric(vectorized_data: np.array) -> np.array:
    tsne_data = TSNE(metric='jaccard').fit_transform(vectorized_data)

    return tsne_data


def average_list(lst):
    if len(lst) is 0:
        return 0

    cum = 0
    for entry in lst:
        cum += float(entry)
    return cum/len(lst)


def parse_results(file_path):
    # for timing [0] is the name, [1] is the amount of noise and [2] is the runtime in s
    # for the teds [0] and [1] as above, [2] is the rted
    file = open(file_path, "r")
    line = file.readline()
    yelp = []
    synth = []
    while line:
        atom = line.rstrip().split(", ")
        if str(Dataset.YELP) == atom[0]:
            yelp.append(atom[1:])
        else:
            synth.append(atom[1:])
        line = file.readline()
        
    for result in [yelp, synth]:
        result_algos = {}
        for entry in result:
            if result_algos.get(entry[0]) is None:
                result_algos[entry[0]] = {entry[3]: {'time': [entry[4]], 'ted': {entry[1]: entry[2]}}}
            elif result_algos.get(entry[0]).get(entry[3]) is None:
                result_algos[entry[0]][entry[3]] = {'time': [entry[4]], 'ted': {entry[1]: entry[2]}}
            else:
                # result_algos[ALGO_NAME][samples]['time'](times)
                # result_algos[ALGO_NAME][samples]['ted'][noise](ted)
                result_algos[entry[0]][entry[3]]['time'].append(entry[4])
                result_algos[entry[0]][entry[3]]['ted'][entry[1]] = entry[2]
        if result is yelp:
            yelp = result_algos
        elif result is synth:
            synth = result_algos
    file.close()

    for result in [synth, yelp]:
        dataset_str = "synth" if result is synth else "yelp"
        for algo in result:
            out_file = open(path.join(BASE, "doc", dataset_str + "_" + algo + "_rt.dat"), "w+")
            for sample in result[algo]:
                result[algo][sample]['time'] = average_list(result[algo][sample]['time'])
                out_file.write(sample + "    " + str(result[algo][sample]['time']) + "\n")
            out_file.close()

    ted_to_files(synth)
    avg_ted_to_files(synth)

    return synth, yelp


def ted_to_files(result_dict):
    for algo in result_dict:
        for sample in result_dict[algo]:
            out_file = open(path.join(BASE, "doc", algo + "_ted" + "_" + str(sample) + ".dat"), "w+")
            for noise_step in result_dict[algo][sample]['ted']:
                out_file.write(noise_step + "    " + result_dict[algo][sample]['ted'][noise_step] + "\n")
            out_file.close()


def avg_ted_to_files(result_dict):
    for algo in result_dict:
        vals = {str(0): [], str(0.05): [], str(0.1): [], str(0.2): [], str(0.33): []}
        for sample in result_dict[algo]:
            for noise_step in result_dict[algo][sample]['ted']:
                vals[noise_step].append(float(result_dict[algo][sample]['ted'][noise_step]) / float(sample))

        out_file = open(path.join(BASE, "doc", algo + "_ted_norm.dat"), "w+")
        for k, v in vals.items():
            out_file.write(k + "    " + str(average_list(v)) + "\n")
        out_file.close()

#
#
# def plot_results():
#     p = path.join(IMG_BASE)
#     if not path.exists(p):
#         makedirs(p)
#
#     synth, yelp = parse_results()
#
#     plt.figure()
#     for algo in synth:
#         print(algo)
#         plt.plot(algo.keys(), algo, label=algo)
#
#     plt.locator_params(axis='x', nbins=len(dataset[0]['ted'].keys()))
#     plt.ylabel('Tree Edit Distance')
#     plt.xlabel('% noise')
#     plt.title("Tree Edit Distance per Noise and Algorithm")
#
#     plt.legend(loc='upper left')
#     plt.savefig(path.join(p, "synth_ted_results.pdf"))
#     plt.clf()

    # plt.figure()
    # for elem in results:
    #     if elem in ['trestle', 'optics', 'single', 'ttsas']:
    #         plt.plot([0.0, 0.1, 0.33, 0.5], results[elem], label=elem)
    #
    # plt.locator_params(axis='x', nbins=4)
    # plt.ylabel('Tree Edit Distance')
    # plt.xlabel('% noise')
    # plt.title("Tree Edit Distance per Noise and Algorithm")
    #
    # plt.legend(loc='upper right')
    # plt.savefig(p + "ted_results_reduced.pdf")
    # plt.clf()

    # for dataset in [synth, yelp]:
    #     safe_str = "synth_" if dataset is synth else "yelp_"
    #     plt.figure()
    #     for name, algo in dataset:
    #         plt.plot(algo['time'].keys(), algo['time'].values, label=name)
    #     plt.locator_params(axis='x', nbins=len(dataset[0]['time']))
    #     plt.ylabel('runime in s')
    #     plt.xlabel('No samples')
    #     plt.title("Runtime per Samples and Algorithm")
    #
    #     plt.legend()
    #     plt.savefig(path.join(p, safe_str + "time_bench.pdf"))
    #     plt.clf()
    # plt.close('all')

    # plt.figure()
    # for elem in times:
    #     if elem in ['rock', 'kmedians', 'affinity_prop', 'kmeans', 'spectral', 'kmedoid', 'dbscan', 'som', 'bsas',
    #                 'mbsas', 'rsl', 'hdbscan']:
    #         continue
    #     plt.plot([27, 2000], times[elem], label=elem)
    # plt.locator_params(axis='x', nbins=2)
    # plt.ylabel('runime in s')
    # plt.xlabel('No samples')
    # plt.title("Runtime per Samples and Algorithm")
    #
    # plt.legend()
    # plt.savefig(p + "time_bench_reduced.pdf")
