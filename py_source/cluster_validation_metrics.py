import numpy as np
import math
import sklearn
from s_dbw import S_Dbw
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances

def calinski_harabasz_index(input_df, tsne_data):
    
    k = len(input_df.topic.unique())
    n = len(input_df)
    wgss = 0
    bgss = 0
    whole_center = np.mean(tsne_data, axis=0)
    for i in range(k):
        cluster_center = np.mean(tsne_data[input_df[input_df.topic == i].apply(lambda x: x.name, axis=1)], axis=0)
        for p in tsne_data[input_df[input_df.topic == i].apply(lambda x: x.name, axis=1)]:
            wgss += math.pow(distance.cosine(cluster_center, p), 2)
        bgss += math.pow(distance.cosine(whole_center, cluster_center), 2)
    seperation = bgss / (k-1)
    compactness = wgss / (n-k)
    return seperation / compactness, compactness, seperation

def silhouette_index(input_df, tsne_data):
    cos_dist = cosine_distances(tsne_data)
    k = len(input_df.topic.unique())
    silhouettes = []
    for i in range(k):
        # Compactness
        within_element_idx = input_df[input_df.topic == i].apply(lambda x: x.name, axis=1).values
        ai = np.sum(cos_dist[within_element_idx][:, within_element_idx], axis=1) / (len(cos_dist[within_element_idx][:, within_element_idx][0]) - 1)
        
        # Seperation
        bi = []
        for j in range(k):
            if i != j:
                between_element_idx = input_df[input_df.topic != i].apply(lambda x: x.name, axis=1).values
                bi.append(np.sum(cos_dist[within_element_idx][:, between_element_idx], axis=1) / (len(cos_dist[within_element_idx][:, between_element_idx][0]) - 1))
        
        silhouettes.append(np.mean((np.min(np.array(bi), axis=0) - ai) / np.max(np.array([ai, np.min(np.array(bi), axis=0)]), axis=0)))
        
    return np.mean(silhouettes)

def sdbw_index(input_df, tsne_data):
    return S_Dbw(tsne_data, input_df.topic, metric='cosine')

def get_validation_metrics(input_df, tsne_data):
    ch, com, sep = calinski_harabasz_index(input_df, tsne_data)
    s = silhouette_index(input_df, tsne_data)
    sdbw = sdbw_index(input_df, tsne_data)

    return [ch, s, sdbw, com, sep]