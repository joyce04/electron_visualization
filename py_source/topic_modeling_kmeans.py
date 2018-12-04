from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def run_kmeans(cluster_K, tsne_data):
    X = normalize(tsne_data, norm='l2')
    kmeans_model = KMeans(n_clusters=cluster_K, init="random", max_iter=30000).fit(X)

    return kmeans_model