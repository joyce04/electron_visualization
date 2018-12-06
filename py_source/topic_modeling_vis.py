import numpy as np
from kmeans_to_pyLDAvis import kmeans_to_prepared_data  

def get_vis_data(X, processed_docs, centers, labels):
    vis_data = kmeans_to_prepared_data(
        X,
        list(set(np.sum(processed_docs.values))),
        centers,
        labels,
        n_printed_words = 10,
        radius = 5
    )

    return vis_data