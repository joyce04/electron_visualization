import umap
from sklearn.feature_extraction.text import CountVectorizer

def run_umap(tsne_data):
    return umap.UMAP().fit_transform(tsne_data)

def get_countvector(processed_docs):
    vect = CountVectorizer()
    vect.fit([' '.join(d) for d in processed_docs])

    tsne_data = vect.transform([' '.join(d) for d in processed_docs]).toarray()

    return tsne_data