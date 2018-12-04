import pandas as pd
import numpy as np
import os
import json
import collections

def get_visualization_json(cluster_K, documents, tsne_result, lda_vis_data, lda_labels, km_vis_data, km_labels, dec_vis_data, dec_labels):

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

    def get_topic_map(lda_docs_by_topic, other_docs_by_topic):
        lda_key_topic_map = {i: [] for i in range(cluster_K)} # key: lda, value: other
        lda_value_topic_map = {i: [] for i in range(cluster_K)} # key: other, value: lda

        for i in range(cluster_K):
            if i not in other_docs_by_topic.index.tolist():
                continue
            lda_topic = int(np.argmax([intersect(other_docs_by_topic.loc[i].values[0], lda_docs_by_topic.loc[j].values[0]) for j in range(cluster_K)]))
            other_topic = i
            lda_key_topic_map[lda_topic].append(other_topic)
            lda_value_topic_map[other_topic].append(lda_topic)

        return lda_key_topic_map, lda_value_topic_map

    def get_hbar_chart_json(vis_data, method):
        hbar_json = {}
        hbar_json['labels'] = vis_data.topic_info.Category.unique().tolist()
        hbar_json['max_width'] = vis_data.topic_info[vis_data.topic_info.Category != 'Default'][['Total']].max()[0] * 1.
        for l in vis_data.topic_info.Category.unique().tolist():
            tmp_df = vis_data.topic_info[vis_data.topic_info.Category == l].sort_values(['Category', 'Freq'], ascending=[True, False])
            tmp_df.Total = tmp_df.Total.apply(lambda x: x * 1.)
            hbar_json[l] = list(tmp_df[['Term', 'Freq', 'Total']].sort_values('Freq', ascending=False).reset_index().to_dict('index').values())

        write_json_to_file('./Visualization/res/%s/hbar_data.json' % method, hbar_json)
        return json.loads(json.dumps(hbar_json, ensure_ascii=False, indent='\t').replace('`', ''))

    def get_scatter_chart_json(doc_result, topic_array, method):
        doc_result['topic'] = topic_array
        doc_result = pd.merge(doc_result, pd.DataFrame(tsne_result, columns=['plot_x', 'plot_y']), left_index=True, right_index=True)

        scatter_json = list(doc_result[['id', 'plot_x', 'plot_y', 'topic']].to_dict('index').values())

        write_json_to_file('./Visualization/res/%s/scatter_data.json' % method, scatter_json)

        if 'data_output' not in os.listdir():
            os.mkdir('data_output')

        doc_result.to_csv(os.path.join(os.getcwd(), 'data_output', '%s.tsv' % method), sep='\t', index_label=False)

        return json.loads(json.dumps(scatter_json, ensure_ascii=False, indent='\t').replace('`', ''))

    def get_merged_table_json():
        lda_result = pd.read_csv('./data_output/lda.tsv', sep='\t')
        km_result = pd.read_csv('./data_output/km.tsv', sep='\t')
        dec_result = pd.read_csv('./data_output/dec.tsv', sep='\t')

        lda_docs_by_topic = lda_result.groupby('topic').agg({'id': 'unique'})
        km_docs_by_topic = km_result.groupby('topic').agg({'id': 'unique'})
        dec_docs_by_topic = dec_result.groupby('topic').agg({'id': 'unique'})

        lda_km_topic_map, km_lda_topic_map = get_topic_map(lda_docs_by_topic, km_docs_by_topic)
        lda_dec_topic_map, dec_lda_topic_map = get_topic_map(lda_docs_by_topic, dec_docs_by_topic)

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
        return  json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', ''))

    lda_hbar_json = get_hbar_chart_json(lda_vis_data, 'lda')
    lda_scatter_json = get_scatter_chart_json(documents, lda_labels, 'lda')

    km_hbar_json = get_hbar_chart_json(km_vis_data, 'km')
    km_scatter_json = get_scatter_chart_json(documents, km_labels, 'km')

    dec_hbar_json = get_hbar_chart_json(dec_vis_data, 'dec')
    dec_scatter_json = get_scatter_chart_json(documents, dec_labels, 'dec')

    document_table_json = get_merged_table_json()   

    return lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json