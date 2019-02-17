# -- coding: utf-8 --

from flask import Flask, request, render_template, jsonify, escape, g
import pandas as pd
import json
import collections
import os
import pickle
import spacy
# import sqlite3

from spacy import displacy
from datetime import datetime
from werkzeug.utils import secure_filename
from topic_modeling import run_topic_modeling, load_topic_modeling

app = Flask(__name__, template_folder='../web/', static_folder='../web')

DATA_FILE_NAME = ''
DATA_FILE_EXTENSION = 'none'

ENTITY_DICT_NAME = ''
ENTITY_DICT_EXTENSION = ''
ENTITY_ALGORITHM = 'flashtext'
ENTITY_HEADER = ['entity_type', 'entity_name']

TARGET_COLUMN = ''
NUM_OF_CLUSTER = 8

TABLE_MAX_WIDTH = 1278

def get_recent_models():
	model_files = [f for f in os.listdir('./model_output') if '.pkl' in f]
	recent_models = []

	for file_name in model_files:
		recent_model_propeties = file_name.split('.')[0].split('_')
		# Filename = {model_name}_{num_of_cluster}_{date}_{seq}.pkl
		if len(recent_model_propeties[-1]) == 8:
			model_name = '_'.join(recent_model_propeties[:-2])
			model_date = recent_model_propeties[-1]
			model_seq = 1
			model_cluster = int(recent_model_propeties[-2])
		else:
			model_name = '_'.join(recent_model_propeties[:-3])
			model_date = recent_model_propeties[-2]
			model_seq = int(recent_model_propeties[-1])
			model_cluster = int(recent_model_propeties[-3])

		recent_models.append([model_name, model_cluster, model_date, model_seq])

	rdf = pd.DataFrame(recent_models, columns=['model_name', 'model_cluster', 'model_date', 'model_seq']).reset_index()
	rdf = rdf.sort_values(['model_date', 'model_name', 'model_seq'], ascending=[False, True, False])
	recent_models = [model_files[i] for i in rdf.index.tolist()] + ['%s(%d, %s, %d)' % tuple(rm[1:]) for rm in rdf.values.tolist()]

	return recent_models

def load_data_file(f):
	if DATA_FILE_EXTENSION == 'csv':
		df = pd.read_csv(f).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (DATA_FILE_NAME, DATA_FILE_EXTENSION)), index=False)
	elif DATA_FILE_EXTENSION == 'tsv':
		df = pd.read_csv(f, sep='\t').reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (DATA_FILE_NAME, DATA_FILE_EXTENSION)), sep='\t', index=False)
	else:
		df = pd.DataFrame(['Not Available'], columns=['Col'])

	return df

def load_entity_dictionary(f):
	global ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM

	ENTITY_DICT_NAME = f.filename.split('.')[0]
	ENTITY_DICT_EXTENSION = f.filename.split('.')[1]

	if ENTITY_DICT_EXTENSION == 'csv':
		df = pd.read_csv(f).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), 'entity_files', '%s.%s' % (DATA_FILE_NAME, ENTITY_DICT_EXTENSION)), index=False)
	elif ENTITY_DICT_EXTENSION == 'tsv':
		df = pd.read_csv(f, sep='\t').reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), 'entity_files', '%s.%s' % (DATA_FILE_NAME, ENTITY_DICT_EXTENSION)), sep='\t', index=False)
	else:
		df = pd.DataFrame(['Not Available'], columns=['Col'])

	return df

def save_model_outputs(outputs):
	if 'model_output' not in os.listdir():
		os.mkdir('model_output')

	todaystr = datetime.today().strftime("%Y%m%d")
	existedResult = len([f for f in os.listdir('./model_output') if '%s_%d_%s' % (DATA_FILE_NAME, NUM_OF_CLUSTER, todaystr) in f])
	resultfname = '%s_%d_%s.pkl' % (DATA_FILE_NAME, NUM_OF_CLUSTER, todaystr) if existedResult == 0 else '%s_%d_%s_%d.pkl' % (DATA_FILE_NAME, NUM_OF_CLUSTER, todaystr, (existedResult+1))

	with open(os.path.join(os.getcwd(), 'model_output', resultfname), 'wb') as f:
		pickle.dump(outputs, f)
	f.close()

def make_jqgrid_data(df):
	col_width = [(df[c].apply(lambda x: len(str(x))).values + [len(c)]).max() for c in df.columns]
	col_width = [cw * 7 + 10 for cw in col_width]
	if sum(col_width) > TABLE_MAX_WIDTH:
		col_width = [0 if cw > TABLE_MAX_WIDTH / len(col_width) else cw for cw in col_width]

	col_width = [(TABLE_MAX_WIDTH - sum(col_width)) / len([cw for cw in col_width if cw == 0]) if cw == 0 else cw for cw in col_width]

	json_data = collections.OrderedDict()
	json_data['rows'] = df.to_dict(orient='records')
	jsonp = json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', ''))

	return col_width, jsonp

@app.route('/')
def index():
	"""
	Rerount to index html
	"""
	return render_template('init.html')

@app.route('/main')
def mainPage():
	global DATA_FILE_NAME, DATA_FILE_EXTENSION
	DATA_FILE_NAME = ''
	DATA_FILE_EXTENSION = 'none'
	recent_models = get_recent_models()

	return render_template('main.html', recent = recent_models, fname = DATA_FILE_NAME, fext = DATA_FILE_EXTENSION, column = [], colw = [], data = '{}', flag='init')

@app.route('/upload_file', methods=['POST'])
def upload_file():
	import pickle
	global DATA_FILE_NAME, DATA_FILE_EXTENSION, TABLE_MAX_WIDTH

	f = request.files['file']
	DATA_FILE_NAME = f.filename.split('.')[0]
	DATA_FILE_EXTENSION = f.filename.split('.')[1]

	if DATA_FILE_EXTENSION in ['csv', 'tsv']:
		df = load_data_file(f)
		col_width, jsonp = make_jqgrid_data(df)

		return render_template('main.html', recent = [], fname = DATA_FILE_NAME, fext = DATA_FILE_EXTENSION, column = df.columns.tolist(), colw = col_width, data = jsonp, flag='upload')
	elif DATA_FILE_EXTENSION == 'pkl':
		saved_model = pickle.load(f)
		with open(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (DATA_FILE_NAME, DATA_FILE_EXTENSION)), 'wb') as f:
			pickle.dump(saved_model, f)

		return render_template('load.html', k = 0, fname = DATA_FILE_NAME, fext = DATA_FILE_EXTENSION, target_column = '')

@app.route('/upload_custom_entity', methods=['POST'])
def upload_custom_entity():
	global TARGET_COLUMN, NUM_OF_CLUSTER
	TARGET_COLUMN = request.form['target_column']
	NUM_OF_CLUSTER = int(request.form['k']) if len(request.form['k']) > 0 else 8

	return render_template('custom_entity.html', column = [], colw = [], data = '{}', flag='init')

@app.route('/upload_dict', methods=['POST'])
def upload_entity_dict():
	df = load_entity_dictionary(f = request.files['file'])
	col_width, jsonp = make_jqgrid_data(df)
	print('upload finished')
	return render_template('custom_entity.html', column = df.columns.tolist(), colw = col_width, data = jsonp, flag='upload')

@app.route('/prepare_model', methods=['POST'])
def prepare_model():
	global ENTITY_DICT_EXTENSION, ENTITY_HEADER, ENTITY_ALGORITHM
	ENTITY_ALGORITHM = request.form['er_algo']
	ENTITY_HEADER = [request.form['entity_type_col'], request.form['entity_term_col']]

	return render_template('load.html', fext = DATA_FILE_EXTENSION)

@app.route('/run_model', methods=['POST'])
def run_model():
	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json = run_topic_modeling(cluster_K = NUM_OF_CLUSTER, file_name=DATA_FILE_NAME, file_extension=DATA_FILE_EXTENSION, target_column_name=TARGET_COLUMN)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json, DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER)
	save_model_outputs(outputs)
	print(DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json,
							km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/import_model', methods=['POST'])
def import_model():
	with open(os.path.join(os.getcwd(), 'data_files', '%s.%s' % (DATA_FILE_NAME, DATA_FILE_EXTENSION)), 'rb') as f:
		saved_model = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json = load_topic_modeling(saved_model)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json, DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER)
	# save_model_outputs(outputs, DATA_FILE_NAME, len(lda_hbar_json['labels'])-1)
	save_model_outputs(outputs)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json,
							km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/load_model/<model_name>', methods=['GET', 'POST'])
def load_model(model_name):
	global DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER

	with open(os.path.join(os.getcwd(), 'model_output', '%s.pkl' % model_name), 'rb') as f:
		outputs = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json, DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER = outputs
	print(DATA_FILE_NAME, DATA_FILE_EXTENSION, ENTITY_DICT_NAME, ENTITY_DICT_EXTENSION, ENTITY_ALGORITHM, ENTITY_HEADER, TARGET_COLUMN, NUM_OF_CLUSTER)
	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/detail', methods=['POST'])
def load_detail():
	barjson = json.loads(request.form['barjson'])
	doctable = json.loads(request.form['doctable'])
	type = json.loads(request.form['typejson'])
	dist = pd.DataFrame(doctable['rows']).groupby('topic_'+type)['document'].count().to_json()

	return render_template('detail.html', bar_json = barjson, document_table_json = doctable, distrib_json = dist, type_json = type)

def initialize_keyword_processors(entity_df):
	from flashtext import KeywordProcessor

	key_pros = {}
	for e, r in entity_df.groupby(ENTITY_HEADER[0]):
		key_pro = KeywordProcessor(case_sensitive=False)
		for v in r.entity_term.values.tolist():
			key_pro.add_keyword(v.strip())
		key_pros[e] = key_pro
	return key_pros

# def check_matching_entities(sub_text, key_procs, is_spacy):
# 	# c_entities = []
# 	ents = []
# 	for ent, pro in key_procs.items():
# 		found = pro.extract_keywords(sub_text)
# 		if is_spacy:
# 			for f in found:

		# else:

# def get_total_matching_num(sub_text, key_procs):
# 	found = {}
# 	for ent, pro in key_procs.items():
# 		found[ent] = pro.extract_keywords(sub_text)
# 	return found

@app.route('/entities', methods=['POST'])
def get_entities():
	import numpy as np
	import random
	from spacy.util import minibatch, compounding

	# spacy.util.set_data_path('./py_source/nlp_data')
	global NLP
	NLP = spacy.load('en')

	type = 'topic_'+request.form['type']
	docdata = json.loads(request.form['data'])
	df_rows = pd.DataFrame(docdata['rows'])
	df_rows['document'] = df_rows.document.apply(lambda x: ' '.join(str(x).split()))
	df_rows['length'] = df_rows.document.apply(lambda x: len(str(x)))

	global KEY_PROS
	if DATA_FILE_NAME !='':
		dic_df = pd.read_csv(os.path.join(os.getcwd(), 'entity_files/', '%s.%s' % (DATA_FILE_NAME, ENTITY_DICT_EXTENSION)))
		dic_df = dic_df[ENTITY_HEADER]
	# print(dic_df)
		KEY_PROS = initialize_keyword_processors(dic_df)
	else:
		print('no entity file')

	ent_list = []

	print(ENTITY_ALGORITHM)

	if DATA_FILE_NAME !='' and ENTITY_ALGORITHM == 'spacy':
		# print(dic_df.loc[dic_df.shape[0]-1])
		train_docs = []

		for idx, row in df_rows.iterrows():
			doc = row['document']

			train_ents = []
			for ent, processor in KEY_PROS.items():
				found = processor.extract_keywords(doc)
				for f in found:
					if doc.find(f)>=0:
						train_ents.append((doc.find(f), doc.find(f)+len(f), ent))
			train_docs.append((doc, {'entities':train_ents}))

		if 'ner' not in NLP.pipe_names:
			ner = NLP.create_pipe('ner')
			NLP.add_pipe(ner)
		else:
			ner = NLP.get_pipe('ner')

		for k in KEY_PROS.keys():
			ner.add_label(k)
		optimizer = NLP.entity.create_optimizer()
		other_pipes = [pipe for pipe in NLP.pipe_names if pipe != 'ner']

		with NLP.disable_pipes(*other_pipes):  # only train NER
			for itn in range(10):
				random.shuffle(train_docs)
				losses = {}

				batches = minibatch(train_docs, size=compounding(4., 32., 1.001))
				for batch in batches:
					texts, annotations = zip(*batch)
					NLP.update(texts, annotations, sgd=optimizer, drop=0.1, losses=losses)
				print('Losses', losses)

		max_len = 90000 if 90000 < df_rows.shape[0] else df_rows.shape[0]
		en_nlp = spacy.load('en')
		for ind, grp in df_rows.groupby(type):
			cum_len = 0
			start_ind = 0

			for idx, row in grp.reset_index().iterrows():
				cum_len += row.length
				if cum_len >= max_len :
					sub_text_grp = grp.loc[start_ind:idx]['document']
					# print(sub_text)
					start_ind =idx
					# print(' '.join(sub_text.tolist()))
					sub_text = ' '.join(sub_text_grp.tolist())

					doc = NLP(sub_text)
					doc_ = en_nlp(sub_text)
					for ent in (doc.ents+doc_.ents):
						if len(ent.text) >2:
							ent_list.append({'cluster':ind, 'text':ent.text, 'label':ent.label_})

	else:
		max_len = 90000 if 90000 < df_rows.shape[0] else df_rows.shape[0]
		for ind, grp in df_rows.groupby(type):
			cum_len = 0
			start_ind = 0

			for idx, row in grp.reset_index().iterrows():
				cum_len += row.length
				if cum_len >= max_len :
					sub_text_grp = grp.loc[start_ind:idx]['document']
					# print(sub_text)
					start_ind =idx
					# print(' '.join(sub_text.tolist()))
					sub_text = ' '.join(sub_text_grp.tolist())

					for ent, pro in KEY_PROS.items():
						found = pro.extract_keywords(sub_text)
						for f in found:
							ent_list.append({'cluster':ind, 'text':'-', 'label':ent})

					doc = NLP(sub_text)
					for ent in doc.ents:
						if len(ent.text) >2:
							ent_list.append({'cluster':ind, 'text':ent.text, 'label':ent.label_})

	print(ent_list)
	if len(ent_list) == 0:
		for ind in range(len(df_rows[type].unique())):
			ent_list.append({'cluster': ind, 'text': np.nan, 'label': np.nan})

	ent_df = pd.DataFrame(ent_list).groupby(['cluster', 'label']).count()
	# print(ent_df)
	json_data = collections.OrderedDict()
	json_data['counts'] = ent_df.reset_index().to_dict(orient='records')
	return json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', '')

@app.route('/entity', methods=['POST'])
def get_entity():
	from spacy.tokens import Span

	target_text = request.form['content']

	doc = NLP(target_text)
	if KEY_PROS is not None:
		ents = []
		for ent, pro in KEY_PROS.items():
			found = pro.extract_keywords(target_text)
			ent_label = doc.vocab.strings.add(ent)
			for f in found:
				# print(f)
				f_arr = f.lower().split(' ')
				# print(f_arr)
				print([i for i in range(len(target_text.split(' ')))])
				f_index = list(filter(lambda x: target_text.lower().split(' ')[x].find(f_arr[0])>=0, [i for i in range(len(target_text.split(' ')))]))[0]
				# print(f_index)
				# print(f_index+len(f_arr))
				# print(ent)
				ent_span = Span(doc, f_index, f_index+len(f_arr), label=ent_label)
				ents.append(ent_span)
		ents.extend(list(doc.ents))
		doc.ents = ents

	dochtml = displacy.render(doc, style='ent')

	return json.dumps({ 'dochtml': dochtml }, ensure_ascii=False, indent='\t')

if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
