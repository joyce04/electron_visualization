def run_topic_modeling(cluster_K = 8, fname='mallet_top_sen', fext='tsv', target_column_name='Origin_Text', train_flag=True):
    # Import Libraries
    import pandas as pd
    import numpy as np
    import pyLDAvis.gensim

    from sklearn.preprocessing import normalize

    from topic_modeling_preprocess import preprocess
    from topic_modeling_tsne import run_tsne
    from topic_modeling_lda import run_lda
    from topic_modeling_kmeans import run_kmeans
    from topic_modeling_dec import run_dec
    from topic_modeling_vis import get_vis_data
    from topic_modeling_visualization import get_visualization_json

    print("Start")

    ## Load Raw Data
    print("Load Data & Preprocessing")
    documents, processed_docs = preprocess(fname, fext, target_column_name)
    tsne_data, tsne_result = run_tsne(processed_docs)
    
    ## LDA
    print("LDA")
    topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency = run_lda(cluster_K, processed_docs)

    lda_data = {
        'topic_term_dists':topic_term_dists,
        'doc_topic_dists':doc_topic_dists,
        'doc_lengths':doc_lengths,
        'vocab':vocab,
        'term_frequency':term_frequency
    }

    lda_vis_data = pyLDAvis.prepare(**lda_data)
    lda_labels = []
    for i in range(len(documents)):
        lda_labels.append(np.argmax(lda_data['doc_topic_dists'][i]))

    print(len(lda_labels), len(documents))

    ## K-Means
    print("K-Means")
    kmeans_model = run_kmeans(cluster_K, tsne_data)
    km_labels = kmeans_model.labels_
    km_vis_data = get_vis_data(normalize(tsne_data, norm='l2'), processed_docs, kmeans_model.cluster_centers_, km_labels)

    ## DEC
    print("DEC")
    dec_labels = run_dec(cluster_K, tsne_data)   

    x_data = pd.DataFrame(tsne_data)
    x_data['y'] = dec_labels
    dec_vis_data = get_vis_data(tsne_data, processed_docs, x_data.groupby('y').mean().reset_index().values[:, 1:], dec_labels)


    ## Visualization
    return get_visualization_json(cluster_K, documents, tsne_result, lda_vis_data, lda_labels, km_vis_data, km_labels, dec_vis_data, dec_labels)

def load_topic_modeling(saved_model):
    # Import Libraries
    import pandas as pd
    import numpy as np
    import pyLDAvis
    import pyLDAvis.gensim as genldavis

    import sklearn
    from sklearn.preprocessing import normalize
    
    from topic_modeling_tsne import run_tsne
    from topic_modeling_vis import get_vis_data
    from topic_modeling_visualization import get_visualization_json

    print("Load Data & Preprocessing")
    documents = saved_model['documents']
    processed_docs = saved_model['processed_docs']
    tsne_data, tsne_result = run_tsne(processed_docs)

    # LDA
    print("LDA")

    lda_result = saved_model['lda_result']
    if len(lda_result) == 3:
        lda_vis_data = genldavis.prepare(**lda_result)
        lda_labels = []
        for i in range(len(documents)):
            lda_labels.append(np.argmax([p for t, p in lda_result['topic_model'][lda_result['corpus'][i]]]))
    elif len(lda_result) == 6:
        lda_vis_data = pyLDAvis.prepare(**lda_result)
        lda_labels = []
        for i in range(len(documents)):
            lda_labels.append(np.argmax(lda_result['doc_topic_dists'][i]))

    # K-Means
    print("K-Means")
    
    kmeans_result = saved_model['kmeans_result']
    if type(kmeans_result) == sklearn.cluster.k_means_.KMeans:
        kmeans_centers = kmeans_result.cluster_centers_
        kmeans_labels = kmeans_result.labels_
    else:
        kmeans_centers = kmeans_result['cluster_centers']
        kmeans_labels = kmeans_result['labels']

    km_vis_data = get_vis_data(normalize(tsne_data, norm='l2'), processed_docs, kmeans_centers, kmeans_labels)


    # DEC
    print("DEC")
    dec_labels = saved_model['dec_result']

    x_data = pd.DataFrame(tsne_data)
    x_data['y'] = dec_labels
    dec_vis_data = get_vis_data(tsne_data, processed_docs, x_data.groupby('y').mean().reset_index().values[:, 1:], dec_labels)

    ## Visualization
    return get_visualization_json(len(np.unique(lda_labels)), documents, tsne_result, lda_vis_data, lda_labels, km_vis_data, kmeans_labels, dec_vis_data, dec_labels)