import pandas as pd

def _df_topic_coordinate(topic_coordinates):
    with open('./data_output/topic_coordinates.tsv', 'w', encoding='utf-8') as f:
        f.write('topic\tx\ty\ttopics\tcluster\tFreq\n')
        for row in topic_coordinates:
            row_strf = '\t'.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./data_output/topic_coordinates.tsv', sep='\t')

def _df_topic_info(topic_info):
    with open('./data_output/topic_info.tsv', 'w', encoding='utf-8') as f:
        f.write('term\tCategory\tFreq\tTerm\tTotal\tloglift\tlogprob\n')
        for row in topic_info:
            row_strf = '\t'.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./data_output/topic_info.tsv', sep='\t')

def _df_token_table(token_table):
    with open('./data_output/token_table.tsv', 'w', encoding='utf-8') as f:
        f.write('term\tTopic\tFreq\tTerm\n')
        for row in token_table:
            row_strf = '\t'.join((str(v) for v in row))
            f.write('%s\n' % row_strf)
    return pd.read_csv('./data_output/token_table.tsv', sep='\t')