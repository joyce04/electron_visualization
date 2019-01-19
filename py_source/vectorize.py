import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

def run_tsne(processed_docs):
    ## T-SNE
    tsne_data = get_countvector(processed_docs)
    tsne_result = TSNE(learning_rate=300, init='pca').fit_transform(np.array(tsne_data))

    return tsne_data, tsne_result

def run_umap(tsne_data):
    return umap.UMAP().fit_transform(tsne_data)

def get_countvector(processed_docs):
    vect = CountVectorizer()
    vect.fit([' '.join(d) for d in processed_docs])

    tsne_data = vect.transform([' '.join(d) for d in processed_docs]).toarray()

    return tsne_data