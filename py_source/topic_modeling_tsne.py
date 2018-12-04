import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

def run_tsne(processed_docs):
    ## T-SNE
    vect = CountVectorizer()
    vect.fit([' '.join(d) for d in processed_docs])

    tsne_data = vect.transform([' '.join(d) for d in processed_docs]).toarray()
    tsne_result = TSNE(learning_rate=300, init='pca').fit_transform(np.array(tsne_data))

    return tsne_data, tsne_result