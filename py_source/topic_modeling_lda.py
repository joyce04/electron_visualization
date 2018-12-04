import pandas as pd
import numpy as np
import random
import operator
from collections import Counter

def get_variables(K, processed_docs):
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
           
def run_lda(n_clusters, processed_docs, itercnt=3):
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

    V, D, document_topic_counts, topic_word_counts, topic_counts, document_lengths, distinct_words = get_variables(n_clusters, processed_docs)

    # 각 단어를 임의의 토픽에 랜덤 배정
    document_topics = [[random.randrange(n_clusters) for word in document] for document in processed_docs]

    # 위와 같이 랜덤 초기화한 상태에서 
    # AB를 구하는 데 필요한 숫자를 세어봄
    for d in range(D):
        for word, topic in zip(processed_docs[d], document_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1

    for _ in range(itercnt):
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

    labels = np.array([max(document_topic_counts[x].items(), key=operator.itemgetter(1))[0] for x in range(D)])

    return topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency, labels