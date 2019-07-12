from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, FeatureAgglomeration
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
from sklearn import metrics
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess, shlex
from pyclustering.samples.definitions import FCPS_SAMPLES

# TODO: BIRCH (?); CURE ROCK; HSyncNet SyncNet SyncSOM;
from pyclustering.cluster import rock
# TODO Trestle, maybe Coweb
import concept_formation

BASE = "/home/fabian/Nextcloud/workspace/uni/8/bachelor/bachelor_project/"
IMG_ID = "5/"
IMG_PATH = BASE + "doc/img/" + IMG_ID
SYNTHETIC_PLAIN_PATH = BASE + "data/synthetic.json"
SYNTHETIC_ALL_PATH = BASE + "data/synthetic_all.json"
SYNTHETIC_BRANCH_PATH = BASE + "data/synthetic_branch.json"
SYNTHETIC_LEVELS_PATH = BASE + "data/synthetic_levels.json"
SYNTHETIC_NAMES_PATH = BASE + "data/synthetic_names.json"
BUSINESSES_PATH = BASE + "data/business.json"
CACHE_PATH = "/tmp/"


def main(dataset: str):
    input = "java -jar synthetic_data_generator.jar -p '/home/fabian/Nextcloud/workspace/uni/8/bachelor_project/data/' -d 3 -w 2 -n 1 1 1 -pr 0.25"
    args = shlex.split(input)
    subprocess.Popen(args)

    # how to get result.tree from dendro
    command_line = "java -jar apted.jar -f ../data/synthetic.tree result.tree"
    args = shlex.split(command_line)
    subprocess.Popen(args)

    if dataset == 'synthetic':
        sets = [open_synthetic(path) for path in [SYNTHETIC_PLAIN_PATH, SYNTHETIC_BRANCH_PATH, SYNTHETIC_LEVELS_PATH, SYNTHETIC_NAMES_PATH, SYNTHETIC_ALL_PATH]]
    elif dataset == 'businesses':
        sets = open_businesses()
    else:
        raise Exception("Specify implemented dataset! synthetic or businesses")

    i = 0
    for labels in sets:
        print("Loading finished, starting vectorization")
        vectorizer = CountVectorizer(binary=True)  # or ordinal
        transformed_data = np.array(vectorizer.fit_transform(labels).toarray())  # .astype(bool, copy=False)

        print("Finished vectorization, starting projection")

        # projected_data_umap = umap.UMAP(metric='jaccard').fit_transform(transformed_data)

        projected_data_tsne = TSNE(metric='jaccard').fit_transform(transformed_data)

        # projected_data_pca = PCA(n_components=0.8, svd_solver='full').fit_transform(transformed_data)

        # projected_data_ica = FastICA().fit_transform(transformed_data)

        # projected_data_feat_agglo = FeatureAgglomeration(n_clusters=5, affinity='jaccard', memory=CACHE_PATH,
        #                                                linkage='single')

        # Clustering on Vectorized data3
        # cluster_rock(projected_data_tsne)

        # cluster_birch(projected_data_tsne)
        # print("Birch finished")

        # print("Projection finished, starting clustering")
        # cluster_agglomerative(projected_data_tsne, projected_data_tsne, i)
        # print("Agglomerative finished")
        # cluster_hdbscan(projected_data_tsne, projected_data_tsne, i)
        #print("HDBScan finished")
        #i += 1

    # cluster_dbscan(transformed_data, projected_data),
    # print("DBScan finished")


    # TODO elbow to find no clusters

    # Clustering on projected data
    # for projected_data in [projected__data_umap, projected_data_tsne, projected_data_pca, projected_data_ica,
    #                       projected_data_feat_agglo]:


    # TODO CURE, Affinity prop, optics, kmeans, mean shift, spectral clustering

    # TODO Spectral Biclustering / Coclustering


def open_synthetic(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]

    return labels


def open_businesses():
    with open(BUSINESSES_PATH, "r") as read_file:
        data = [json.loads(line) for line in read_file]

    labels = [entry['categories'] for entry in data]

    labels = [label for label in labels if label is not None]

    return [labels]


def cluster_agglomerative(transformed_data, projected_data, i):
    agglo = AgglomerativeClustering(n_clusters=25, affinity='jaccard', memory=CACHE_PATH,
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
    plt.savefig(IMG_PATH + str(i) + "agglo_dendro.svg")

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
    plt.savefig(IMG_PATH + str(i) + "agglo_clust.svg")
    plt.show()


def cluster_dbscan(transformed_data, projected_data):
    db = DBSCAN(eps=0.99, metric='jaccard', min_samples=40, n_jobs=-1)
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


def cluster_hdbscan(transformed_data, projected_data, i):
    hdb = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=24, memory=CACHE_PATH, core_dist_n_jobs=-1) \
        .fit(transformed_data)
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
    plt.savefig(IMG_PATH + str(i) + "hdbscan_clusters.svg")
    plt.show()
    plt.figure()
    hdb.single_linkage_tree_.plot()
    plt.savefig(IMG_PATH + str(i) + "hdbscan_dendro.svg")
    plt.show()
    plt.figure()
    hdb.condensed_tree_.plot()
    plt.savefig(IMG_PATH + str(i) + "hdbscan_condensed.svg")
    plt.show()


def cluster_rock(transformed_data):
    rock_inst = rock.rock(transformed_data, 2, 5)
    rock_inst.process()
    clusters = rock_inst.get_clusters()
    for cluster in clusters:
        print(cluster)


# Does not work with bool data, as the measures get all 0. returns (1, 780)?! actually there are 625 nodes
def cluster_birch(transformed_data):
    transformed_data = np.transpose(transformed_data)
    agglo = AgglomerativeClustering(n_clusters=5, affinity='jaccard', memory=CACHE_PATH,
                                    linkage='single')
    birch = Birch(threshold=0.0000001, n_clusters=agglo).fit(transformed_data)
    print(birch)

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
    dendrogram(linkage_matrix, labels=birch.labels_)
    plt.savefig(IMG_PATH + "agglo_dendro.svg")

    plt.figure()
    labels = birch.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = transformed_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

    plt.title('BIRCH + Agglo: Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(IMG_PATH + "agglo_clust.svg")
    plt.show()


if __name__ == '__main__':
    # 'synthetic' or 'businesses'
    main('synthetic')
