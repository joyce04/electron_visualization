# Import Libraries
import pandas as pd
import numpy as np
import operator
import random
import json
import collections
import os

from collections import Counter
from kmeans_to_pyLDAvis import kmeans_to_prepared_data

import gensim
import pyLDAvis.gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn import metrics

from keras.models import Model
from keras import backend as K
from keras import layers
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.engine.topology import Layer, InputSpec

def run_topic_modeling(fname='mallet_top_sen', fext='tsv', target_column_name='Origin_Text', train_flag=True):
    # Declare Class
    class ClusteringLayer(Layer):
        """
        Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
        sample belonging to each cluster. The probability is calculated with student's t-distribution.

        # Example
        ```
            model.add(ClusteringLayer(n_clusters=10))
        ```
        # Arguments
            n_clusters: number of clusters.
            weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
            alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
        # Input shape
            2D tensor with shape: `(n_samples, n_features)`.
        # Output shape
            2D tensor with shape: `(n_samples, n_clusters)`.
        """

        def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super(ClusteringLayer, self).__init__(**kwargs)
            self.n_clusters = n_clusters
            self.alpha = alpha
            self.initial_weights = weights
            self.input_spec = InputSpec(ndim=2)

        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]
            self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
            self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            self.built = True

        def call(self, inputs, **kwargs):
            """ student t-distribution, as same as used in t-SNE algorithm.
            Measure the similarity between embedded point z_i and centroid µ_j.
                    q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                    q_ij can be interpreted as the probability of assigning sample i to cluster j.
                    (i.e., a soft assignment)
            Arguments:
                inputs: the variable containing data, shape=(n_samples, n_features)
            Return:
                q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
            """
            q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
            q **= (self.alpha + 1.0) / 2.0
            q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
            return q

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            return input_shape[0], self.n_clusters

        def get_config(self):
            config = {'n_clusters': self.n_clusters}
            base_config = super(ClusteringLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


    # Declare Functions
    def lemmatize_stemming(text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    def get_variables(K):
        # 사용자가 원하는 토픽의 갯수
        K = 8

        # 각 토픽이 각 문서에 할당되는 횟수
        # Counter로 구성된 리스트
        # 각 Counter는 각 문서를 의미
        document_topic_counts = [Counter() for _ in processed_docs]

        # 각 단어가 각 토픽에 할당되는 횟수
        # Counter로 구성된 리스트
        # 각 Counter는 각 토픽을 의미
        topic_word_counts = [Counter() for _ in range(K)]

        # 각 토픽에 할당되는 총 단어수
        # 숫자로 구성된 리스트
        # 각각의 숫자는 각 토픽을 의미함
        topic_counts = [0 for _ in range(K)]

        # 각 문서에 포함되는 총 단어수
        # 숫자로 구성된 리스트
        # 각각의 숫자는 각 문서를 의미함
        document_lengths = list(map(len, processed_docs))

        # 단어 종류의 수
        distinct_words = set(word for document in processed_docs for word in document)
        V = len(distinct_words)

        # 총 문서의 수
        D = len(processed_docs)

        return V, D, document_topic_counts, topic_word_counts, topic_counts, document_lengths, distinct_words

    def p_topic_given_document(topic, d, n_clusters, alpha=0.1):
        # 문서 d의 모든 단어 가운데 topic에 속하는
        # 단어의 비율 (alpha를 더해 smoothing)
        return ((document_topic_counts[d][topic] + alpha) /
                (document_lengths[d] + n_clusters * alpha))

    def p_word_given_topic(word, topic, beta=0.1):
        # topic에 속한 단어 가운데 word의 비율
        # (beta를 더해 smoothing)
        return ((topic_word_counts[topic][word] + beta) /
                (topic_counts[topic] + V * beta))

    def topic_weight(d, word, k, n_clusters):
        # 문서와 문서의 단어가 주어지면
        # k번째 토픽의 weight를 반환
        return p_word_given_topic(word, k) * p_topic_given_document(k, d, n_clusters)

    def choose_new_topic(d, word, n_clusters):
        return sample_from([topic_weight(d, word, k, n_clusters) for k in range(n_clusters)])

    def sample_from(weights):
        # i를 weights[i] / sum(weights)
        # 확률로 반환
        total = sum(weights)
        # 0과 total 사이를 균일하게 선택
        rnd = total * random.random()
        # 아래 식을 만족하는 가장 작은 i를 반환
        # weights[0] + ... + weights[i] >= rnd
        for i, w in enumerate(weights):
            rnd -= w
            if rnd <= 0:
                return i

    def autoencoderConv2D_1(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
        input_img = Input(shape=input_shape)
        if input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'
        x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

        x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

        x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)

        x = Flatten()(x)
        encoded = Dense(units=filters[3], name='embedding')(x)
        x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

        x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
        x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

        x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

        decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

    def autoencoder(dims, act='relu', init='glorot_uniform'):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

        # hidden layer
        encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def intersect(a, b):
        return len(list(set(a) & set(b)))

    def write_json_to_file(path, json_str):
        pwd = os.getcwd()
        pathList = path.split('/')
        for p in pathList[:-1]:
            if p not in os.listdir() and p != '.':
                os.mkdir(p)
            os.chdir('./%s' % p)
        os.chdir(pwd)

        if '.' in pathList:
            pathList = [p for p in pathList if p != '.']

        f = open(os.path.join(*[os.getcwd()] + pathList), "w")
        f.write(json.dumps(json_str, ensure_ascii=False, indent='\t'))
        f.close()

    def get_jsonp(json_str):
        return json.loads(json.dumps(json_str, ensure_ascii=False, indent='\t').replace('`', ''))

    def get_hbar_chart_json(vis_data, method):
        hbar_json = {}
        hbar_json['labels'] = vis_data.topic_info.Category.unique().tolist()
        hbar_json['max_width'] = vis_data.topic_info[vis_data.topic_info.Category != 'Default'][['Total']].max()[0] * 1.
        for l in vis_data.topic_info.Category.unique().tolist():
            tmp_df = vis_data.topic_info[vis_data.topic_info.Category == l].sort_values(['Category', 'Freq'], ascending=[True, False]).groupby('Category').head()
            tmp_df.Total = tmp_df.Total.apply(lambda x: x * 1.)
            hbar_json[l] = list(tmp_df[['Term', 'Freq', 'Total']].sort_values('Freq', ascending=False).reset_index().to_dict('index').values())

        write_json_to_file('./Visualization/res/%s/hbar_data.json' % method, hbar_json)
        return get_jsonp(hbar_json)

    def get_scatter_chart_json(topic_array, method):
        doc_result = documents[['index', 'Origin_Text']]
        doc_result.columns = ['id', 'document']
        doc_result['topic'] = topic_array
        doc_result = pd.merge(doc_result, pd.DataFrame(tsne_result, columns=['plot_x', 'plot_y']), left_index=True, right_index=True)

        scatter_json = list(doc_result[['id', 'plot_x', 'plot_y', 'topic']].to_dict('index').values())

        write_json_to_file('./Visualization/res/%s/scatter_data.json' % method, scatter_json)

        if 'data_output' not in os.listdir():
            os.mkdir('data_output')

        doc_result.to_csv(os.path.join(os.getcwd(), 'data_output', '%s.tsv' % method), sep='\t', index_label=False)

        return get_jsonp(scatter_json)


    print("Start")

    np.random.seed(2018)
    random.seed(2018)
    nltk.download('wordnet')
    
    ## Load Raw Data
    print("Load Data & Preprocessing")

    if fext == 'csv':
        origin_data = pd.read_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), index_col=0).reset_index(drop=True)
    elif fext == 'tsv':
        origin_data = pd.read_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), sep='\t', index_col=0).reset_index(drop=True)
    else:
        origin_data = pd.read_csv(os.path.join(os.getcwd(), 'mallet_top_sen.tsv'), sep='\t', index_col=0).reset_index(drop=True)

    train_flag = False

    ## Extract target data
    documents = origin_data[[target_column_name]].reset_index()

    ## Preprocess
    processed_docs = documents[target_column_name].map(preprocess)

    ## T-SNE
    vect = CountVectorizer()
    vect.fit([' '.join(d) for d in processed_docs])

    tsne_data = vect.transform([' '.join(d) for d in processed_docs]).toarray()
    tsne_result = TSNE(learning_rate=300, init='pca').fit_transform(np.array(tsne_data))





    ## LDA
    print("LDA")
    n_clusters = 8
    V, D, document_topic_counts, topic_word_counts, topic_counts, document_lengths, distinct_words = get_variables(n_clusters)

    # 각 단어를 임의의 토픽에 랜덤 배정
    document_topics = [[random.randrange(n_clusters) for word in document] for document in processed_docs]

    # 위와 같이 랜덤 초기화한 상태에서 
    # AB를 구하는 데 필요한 숫자를 세어봄
    for d in range(D):
        for word, topic in zip(processed_docs[d], document_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1

    for iter in range(3):
        for d in range(D):
            for i, (word, topic) in enumerate(zip(processed_docs[d], document_topics[d])):
                # 깁스 샘플링 수행을 위해
                # 샘플링 대상 word와 topic을 제외하고 세어봄
                document_topic_counts[d][topic] -= 1
                topic_word_counts[topic][word] -= 1
                topic_counts[topic] -= 1
                document_lengths[d] -= 1

                # 깁스 샘플링 대상 word와 topic을 제외한 
                # 말뭉치 모든 word의 topic 정보를 토대로
                # 샘플링 대상 word의 새로운 topic을 선택
                new_topic = choose_new_topic(d, word, n_clusters)
                document_topics[d][i] = new_topic

                # 샘플링 대상 word의 새로운 topic을 반영해 
                # 말뭉치 정보 업데이트
                document_topic_counts[d][new_topic] += 1
                topic_word_counts[new_topic][word] += 1
                topic_counts[new_topic] += 1
                document_lengths[d] += 1
        
    topic_term_dists = np.array([topic_word_counts[i][k] for i in range(n_clusters) for k in list(distinct_words)]).reshape((n_clusters, len(distinct_words))) 
    doc_topic_dists = pd.DataFrame([d.values() for d in document_topic_counts]).fillna(0).values
    doc_lengths = np.array(document_lengths)
    vocab = list(distinct_words)
    term_frequency = np.array([topic_word_counts[i][k] for i in range(n_clusters) for k in list(distinct_words)]).reshape((n_clusters, len(distinct_words))).sum(axis=0)

    lda_mallet_data = {
        'topic_term_dists':topic_term_dists,
        'doc_topic_dists':doc_topic_dists,
        'doc_lengths':doc_lengths,
        'vocab':vocab,
        'term_frequency':term_frequency
    }

    lda_vis_data = pyLDAvis.prepare(**lda_mallet_data)

    ## Visualization
    lda_hbar_json = get_hbar_chart_json(lda_vis_data, 'lda')
    lda_scatter_json = get_scatter_chart_json(np.array([max(document_topic_counts[x].items(), key=operator.itemgetter(1))[0] for x in range(D)]), 'lda')




    ## K-Means
    print("K-Means")
    docs = list(documents.Origin_Text.values)
    X = normalize(tsne_data, norm='l2')
    kmeans_model = KMeans(n_clusters=8, init="random", max_iter=30000).fit(X)
    labels = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_

    doc_lengths = np.asarray(X.sum(axis=1)).reshape(-1)
    term_frequency = np.asarray(X.sum(axis=0)).reshape(-1)
    docwords = list(processed_docs.apply(lambda x: len(x)).values)
    vocab = list(set(np.sum(processed_docs.values)))

    km_vis_data = kmeans_to_prepared_data(
        X,
        vocab,
        centers,
        labels,
        n_printed_words = 10,
        radius = 5
    )

    ## Visualization
    km_hbar_json = get_hbar_chart_json(km_vis_data, 'km')
    km_scatter_json = get_scatter_chart_json(kmeans_model.labels_, 'km')




    ## DEC
    print("DEC")
    max_count = max([np.max(tsne_data[i]) for i in tsne_data]) * 1.
    x = np.divide(tsne_data, max_count)
    n_clusters = 8

    ## Create Fully Connection Model
    dims = [x.shape[-1], 500, 500, 2000, 10]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 300
    batch_size = 256

    autoencoder, encoder = autoencoder(dims, init=init)
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

    ## Train Model
    if train_flag == True:
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
        autoencoder.save_weights(os.path.join(os.getcwd(), 'data_output', 'drug_ae_weights.h5'))
    else:
        autoencoder.load_weights(os.path.join(os.getcwd(), 'data_output', 'drug_ae_weights.h5'))

    ## Initialize cluster centers using k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    y_pred_last = np.copy(y_pred)

    ## Deep Clustering Train
    loss = 0
    index = 0
    maxiter = 8000
    update_interval = 140
    index_array = np.arange(x.shape[0])

    tol = 0.001 # tolerance threshold to stop training

    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)

    y = None

    if train_flag == True:
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _  = model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.accuracy_score(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        model.save_weights(os.path.join(os.getcwd(), 'data_output', 'drug_DEC_model_final.h5'))
    else:
        model.load_weights(os.path.join(os.getcwd(), 'data_output', 'drug_DEC_model_final.h5'))

    q, _ = model.predict(x, verbose=0)
    p = target_distribution(q)

    y_pred = q.argmax(1)

    doc_lengths = np.asarray(tsne_data.sum(axis=1)).reshape(-1)
    term_frequency = np.asarray(tsne_data.sum(axis=0)).reshape(-1)
    docwords = list(processed_docs.apply(lambda x: len(x)).values)
    vocab = list(set(np.sum(processed_docs.values)))
    x_data = pd.DataFrame(tsne_data)
    x_data['y'] = y_pred
    centers = x_data.groupby('y').mean().reset_index().values[:, 1:]

    dec_vis_data = kmeans_to_prepared_data(
        tsne_data,
        vocab,
        centers,
        y_pred,
        n_printed_words = 10,
        radius = 5
    )

    ## Visualization
    dec_hbar_json = get_hbar_chart_json(dec_vis_data, 'dec')
    dec_scatter_json = get_scatter_chart_json(y_pred, 'dec')




    ## Create Table with 3 Model Results
    print("Data Merge")
    lda_result = pd.read_csv('./data_output/lda.tsv', sep='\t')
    km_result = pd.read_csv('./data_output/km.tsv', sep='\t')
    dec_result = pd.read_csv('./data_output/dec.tsv', sep='\t')

    lda_docs_by_topic = lda_result.groupby('topic').agg({'id': 'unique'})
    km_docs_by_topic = km_result.groupby('topic').agg({'id': 'unique'})
    dec_docs_by_topic = dec_result.groupby('topic').agg({'id': 'unique'})

    lda_km_topic_map = {i: [] for i in range(8)} # key: lda, value: km
    km_lda_topic_map = {i: [] for i in range(8)} # key: km, value: lda

    for i in range(8):
        lda_topic = int(np.argmax([intersect(km_docs_by_topic.iloc[i].values[0], lda_docs_by_topic.iloc[j].values[0]) for j in range(8)]))
        km_topic = i
        lda_km_topic_map[lda_topic].append(km_topic)
        km_lda_topic_map[km_topic].append(lda_topic)

    lda_dec_topic_map = {i: [] for i in range(8)} # key: lda, value: km
    dec_lda_topic_map = {i: [] for i in range(8)} # key: km, value: lda

    for i in range(8):
        lda_topic = int(np.argmax([intersect(dec_docs_by_topic.iloc[i].values[0], lda_docs_by_topic.iloc[j].values[0]) for j in range(8)]))
        dec_topic = i
        lda_dec_topic_map[lda_topic].append(dec_topic)
        dec_lda_topic_map[dec_topic].append(lda_topic)

    merged_result = pd.merge(lda_result[['id', 'document', 'topic']], km_result[['id', 'topic']], on='id', suffixes=('_lda', '_km'))
    merged_result = pd.merge(merged_result[['id', 'document', 'topic_lda', 'topic_km']], dec_result[['id', 'topic']], on='id')
    merged_result.columns = ['id', 'document', 'topic_lda', 'topic_km', 'topic_dec']

    json_data = collections.OrderedDict()
    json_data['lda_km_topic_map'] = lda_km_topic_map
    json_data['km_lda_topic_map'] = km_lda_topic_map
    json_data['lda_dec_topic_map'] = lda_dec_topic_map
    json_data['dec_lda_topic_map'] = dec_lda_topic_map
    json_data['rows'] = merged_result[['document', 'topic_lda', 'topic_km', 'topic_dec']].to_dict(orient='records')

    write_json_to_file('./Visualization/res/document_table.json', json_data)
    document_table_json = get_jsonp(json_data)

    return lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json