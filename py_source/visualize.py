import pandas as pd
import numpy as np
import os
import json
import collections
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

# def get_visualization_json(cluster_K, documents, tsne_result, lda_vis_data, lda_labels, km_vis_data, km_labels, dec_vis_data, dec_labels):
def get_visualization_json(cluster_K, documents, tsne_result, lda_vis_data, lda_labels, lda_metrics, km_vis_data, km_labels, km_metrics, dec_vis_data, dec_labels, dec_metrics):
    # def intersect(a, b):
    #     return len(list(set(a) & set(b)))

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

        f = open(os.path.join(*[os.getcwd()] + pathList), "w", encoding='utf8')
        f.write(json.dumps(json_str, ensure_ascii=False, indent='\t'))
        f.close()

    def get_topic_map(doc_df, col_name):
        lda_value_topic_map = {} # key: other, value: lda

        other_doc_count = doc_df.groupby(col_name).document.count().to_dict()
        other_lda_df = doc_df.groupby([col_name, 'topic_lda']).count().reset_index()
        other_lda_df['doc_ratio'] = other_lda_df.apply(lambda x: x.document / other_doc_count[x[col_name]], axis=1)
        other_lda_df['max_doc_ratio'] = other_lda_df.groupby(col_name)['doc_ratio'].transform('max')
        other_lda_df['doc_ratio_rank'] = other_lda_df.groupby(col_name).doc_ratio.rank(ascending=False)

        for i in range(1, cluster_K+1, 1):
            for idx, row in other_lda_df[other_lda_df.doc_ratio_rank == i].sort_values('doc_ratio', ascending=False).iterrows():
                if int(row.topic_lda) not in lda_value_topic_map.values() and int(row[col_name]) not in lda_value_topic_map.keys():
                    lda_value_topic_map[int(row[col_name])] = int(row.topic_lda)
            if len(lda_value_topic_map) == cluster_K:
                break;

        if len(lda_value_topic_map) != cluster_K:
            remain_lda = list(set(range(cluster_K)) - set(lda_value_topic_map.values()))
            remain_other = list(set(range(cluster_K)) - set(lda_value_topic_map.keys()))

            for i, k in enumerate(remain_other):
                lda_value_topic_map[k] = remain_lda[i]

        return lda_value_topic_map

    def get_hbar_chart_json(vis_data, method, mapper):
        hbar_json = {}
        hbar_json['labels'] = vis_data.topic_info.Category.unique().tolist()
        hbar_json['max_width'] = vis_data.topic_info[vis_data.topic_info.Category != 'Default'][['Total']].max()[0] * 1.
        for l in vis_data.topic_info.Category.unique().tolist():
            real_topic = l if mapper == None or l == 'Default' else 'Topic%s' % str(mapper[int(l.replace('Topic', ''))-1]+1)
            tmp_df = vis_data.topic_info[vis_data.topic_info.Category == l].sort_values(['Category', 'Freq'], ascending=[True, False])
            tmp_df.Total = tmp_df.Total.apply(lambda x: x * 1.)
            hbar_json[real_topic] = list(tmp_df[['Term', 'Freq', 'Total']].sort_values('Freq', ascending=False).reset_index().to_dict('index').values())

        write_json_to_file('./Visualization/res/%s/hbar_data.json' % method, hbar_json)
        return json.loads(json.dumps(hbar_json, ensure_ascii=False, indent='\t').replace('`', ''))

    def get_scatter_chart_json(doc_result, method):
        doc_result = pd.merge(doc_result, pd.DataFrame(tsne_result, columns=['plot_x', 'plot_y']), left_index=True, right_index=True)
        doc_df = doc_result[['id', 'plot_x', 'plot_y', 'topic_%s' % method]]
        doc_df.columns = ['id', 'plot_x', 'plot_y', 'topic']

        scatter_json = list(doc_df.to_dict('index').values())
        
        write_json_to_file('./Visualization/res/%s/scatter_data.json' % method, scatter_json)

        return json.loads(json.dumps(scatter_json, ensure_ascii=False, indent='\t').replace('`', ''))

    def get_merged_table_json(document, lda_labels, km_labels, dec_labels):
        merged_result = document[['id', 'document']]
        merged_result['topic_lda'] = lda_labels
        merged_result['topic_km'] = km_labels
        merged_result['topic_dec'] = dec_labels
        merged_result.columns = ['id', 'document', 'topic_lda', 'topic_km', 'topic_dec']

        km_lda_topic_map = get_topic_map(pd.DataFrame(merged_result[['document', 'topic_lda', 'topic_km', 'topic_dec']]), 'topic_km')
        dec_lda_topic_map = get_topic_map(pd.DataFrame(merged_result[['document', 'topic_lda', 'topic_km', 'topic_dec']]), 'topic_dec')

        merged_result['topic_km'] = merged_result.topic_km.apply(lambda x: km_lda_topic_map[x])
        merged_result['topic_dec'] = merged_result.topic_dec.apply(lambda x: dec_lda_topic_map[x])

        json_data = collections.OrderedDict()
        json_data['rows'] = merged_result[['document', 'topic_lda', 'topic_km', 'topic_dec']].to_dict(orient='records')
        
        write_json_to_file('./Visualization/res/document_table.json', json_data)
        return  json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', '')), merged_result, km_lda_topic_map, dec_lda_topic_map

    def get_metrics_json(lda_metrics, km_metrics, dec_metrics):
        metric_labels = ['Calinski-Harabasz', 'Silhouette', 'S_Dbw', 'Compactness', 'Seperation']

        json_data = collections.OrderedDict()
        json_data['labels'] = metric_labels

        for i, m in enumerate(metric_labels):
            json_data[m] = [lda_metrics[i], km_metrics[i], dec_metrics[i]]

        write_json_to_file('./Visualization/res/metrics.json', json_data)
        return json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', ''))

    document_table_json, mapped_documents, km_lda_topic_map, dec_lda_topic_map = get_merged_table_json(documents, lda_labels, km_labels, dec_labels)

    lda_hbar_json = get_hbar_chart_json(lda_vis_data, 'lda', None)
    lda_scatter_json = get_scatter_chart_json(mapped_documents, 'lda')

    km_hbar_json = get_hbar_chart_json(km_vis_data, 'km', km_lda_topic_map)
    km_scatter_json = get_scatter_chart_json(mapped_documents, 'km')

    dec_hbar_json = get_hbar_chart_json(dec_vis_data, 'dec', dec_lda_topic_map)
    dec_scatter_json = get_scatter_chart_json(mapped_documents, 'dec')

    metrics_json = get_metrics_json(lda_metrics, km_metrics, dec_metrics)

    return lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json
