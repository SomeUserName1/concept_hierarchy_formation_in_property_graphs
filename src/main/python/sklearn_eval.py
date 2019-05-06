from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn import metrics
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import json

BASE = "/home/fabian/Nextcloud/workspace/uni/8/bachelor/bachelor_project/"
IMG_ID = "1/"
IMG_PATH = BASE + "doc/img/" + IMG_ID
SYNTHETIC_PATH = BASE + "data/synthetic.json"
BUSINESSES_PATH = BASE + "data/business.json"


def main(dataset: str):
    if dataset == 'synthetic':
        labels = open_synthetic()
    elif dataset == 'businesses':
        labels = open_businesses()
    else:
        raise Exception("Specify implemented dataset! synthetic or businesses")

    print("Loading finished, starting vectorization")
    vectorizer = CountVectorizer(binary=True)
    transformed_data = vectorizer.fit_transform(labels).toarray()

    print("Finished vectorization, starting projection")

    projected_data = umap.UMAP(metric='jaccard', random_state=42, n_neighbors=30, verbose=True, min_dist=0.0)\
        .fit_transform(transformed_data)

    print("Projection finished, starting clustering")
    cluster_agglomerative(transformed_data, projected_data)
    cluster_dbscan(transformed_data, projected_data)
    cluster_hdbscan(transformed_data, projected_data)


def open_synthetic():
    with open(SYNTHETIC_PATH, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]

    return labels


def open_businesses():
    with open(BUSINESSES_PATH, "r") as read_file:
        data = [json.loads(line) for line in read_file]

    labels = [entry['categories'] for entry in data]

    labels = [label for label in labels if label is not None]

    return labels


def cluster_agglomerative(transformed_data, projected_data):
    agglo = AgglomerativeClustering(n_clusters=5, affinity='jaccard',
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
    plt.savefig(IMG_PATH + "agglo_dendro.svg")

    plt.figure()
    labels = agglo.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = projected_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

    plt.title('Agglomerative: Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(IMG_PATH + "agglo_clust.svg")
    plt.show()


def cluster_dbscan(transformed_data, projected_data):
    db = DBSCAN(eps=0.99, metric='jaccard', min_samples=40)
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

        xy = projected_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = projected_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(IMG_PATH + "dbscan_clusters.svg")
    plt.show()


def cluster_hdbscan(transformed_data, projected_data):
    hdb = hdbscan.HDBSCAN(metric='jaccard', min_cluster_size=40).fit(transformed_data)
    labels = hdb.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(transformed_data, labels))
    print("Cluster  stability per cluster:")
    for val in hdb.cluster_persistence_:
        print("\t %0.3f" % val)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = projected_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = projected_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('HDBSCAN*: Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(IMG_PATH + "hdbscan_clusters.svg")
    plt.show()
    plt.figure()
    hdb.single_linkage_tree_.plot()
    plt.savefig(IMG_PATH + "hdbscan_dendro.svg")
    plt.show()
    plt.figure()
    hdb.condensed_tree_.plot()
    plt.savefig(IMG_PATH + "hdbscan_condensed.svg")
    plt.show()


if __name__ == '__main__':
    # 'synthetic' or 'businesses'
    main('businesses')
