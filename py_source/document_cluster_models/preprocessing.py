import pandas as pd
import numpy as np
import os
import pyLDAvis.gensim as gensim
import random
import nltk
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer

def lemmatize_stemming(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(token, pos='v')))
    return result

def preprocess(file_name, file_extension, target_column_name):

    np.random.seed(2018)
    random.seed(2018)
    nltk.download('wordnet')

    if file_extension == 'csv':
        origin_data = pd.read_csv(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (file_name, file_extension)))
    elif file_extension == 'tsv':
        origin_data = pd.read_csv(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (file_name, file_extension)), sep='\t')
    else:
        origin_data = pd.read_csv(os.path.join(os.getcwd(), 'data_files', 'mallet_top_sen.tsv'), sep='\t')

    # train_flag = False

    ## Extract target data
    documents = origin_data[[target_column_name]].reset_index()
    documents = documents[['index', target_column_name]]
    documents.columns = ['id', 'document']
    documents.document = documents.document.astype('str')

    ## Preprocess
    processed_docs = documents['document'].map(lemmatize_stemming)

    return documents, processed_docs
