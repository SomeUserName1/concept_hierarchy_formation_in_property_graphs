from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score
import hdbscan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


def main(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]

    vectorizer = CountVectorizer(binary=True)
    transformed_data = vectorizer.fit_transform(labels).toarray()

    # FIXME plotting, hdbscan plots

    cluster_agglomerative(transformed_data)
    # cluster_dbscan(transformed_data)
    cluster_hdbscan(transformed_data)


def cluster_agglomerative(transformed_data):
    agglo = AgglomerativeClustering(n_clusters=4, affinity='jaccard',
                                    linkage='single')
    agglo.fit(transformed_data)

    plt.title('Hierarchical Clustering Dendrogram')

    children = agglo.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=agglo.labels_)

    labels = agglo.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
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

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def cluster_dbscan(transformed_data):
    db = DBSCAN(eps=0.9, metric='jaccard', min_samples=4)
    db.fit(transformed_data)
    # DBSCAN Plotting and measures
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(transformed_data, labels))

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = transformed_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = transformed_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def cluster_hdbscan(transformed_data):
    arr = np.array(transformed_data)
    print(arr.shape)
    hdb = hdbscan.HDBSCAN(metric='jaccard')
    hdb.fit(transformed_data)
    # color_palette = sns.color_palette('deep', 625)
    # cluster_colors = [color_palette[x] if x >= 0
    #                   else (0.5, 0.5, 0.5)
    #                   for x in hdb.labels_]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                          zip(cluster_colors, hdb.probabilities_)]
    # plt.scatter(arr, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    hdb.single_linkage_tree_.plot()
    hdb.condensed_tree_.plot()


if __name__ == '__main__':
    PATH = "/home/someusername/snap/nextcloud-client/10/Nextcloud/workspace/uni/8/bachelor/bachelor_project/data" \
           "/synthetic.json"
    main(PATH)
