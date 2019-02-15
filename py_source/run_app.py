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

@app.route('/')
def index():
	"""
	Rerount to index html
	"""
	return render_template('init.html')

@app.route('/main')
def mainPage():
	model_files = [f for f in os.listdir('./model_output') if 'pkl' in f]
	recent_models = []

	for fn in [f for f in os.listdir('./model_output') if '.pkl' in f]:
		recent_model_propeties = fn.split('.')[0].split('_')
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
	rdf = rdf.sort_values(['model_date', 'model_name', 'model_seq'], ascending=[False, True, True])
	recent_models = [model_files[i] for i in rdf.index.tolist()] + ['%s(%d, %s, %d)' % tuple(rm[1:]) for rm in rdf.values.tolist()]

	return render_template('main.html', recent = recent_models, fname = '', fext='none', column = [], colw = [], data = '{}', flag='init')

@app.route('/user/<name>', methods=['GET', 'POST'])
def user(name):
	"""
	Move to user.html with given user name

	Args:
		name (str) : user name
	Returns:
		html : user.html
	"""
	return render_template('user.html', name=name)

@app.route('/d3/<chart_type>', methods=['GET', 'POST'])
def open_chart(chart_type):
	"""
	Move to chart.html by given chart type

	Args:
		chart_type (str) : chart type
	Returns:
		html : chart.html
	"""
	if chart_type == 'radar_chart':
		return render_template('radar.html')
	else:
		return render_template('bar_chart.html')

@app.route('/grid', methods=['GET', 'POST'])
def open_grid():
	return render_template('grid.html', fname = '', fext='none', column = [], colw = [], data = '{}', flag='init')

@app.route('/upload_file', methods=['POST'])
def upload_file():
	import pickle

	f = request.files['file']
	fname = f.filename.split('.')[0]
	fext = f.filename.split('.')[1]

	if fext == 'csv':
		df = pd.read_csv(f).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), index=False)
	elif fext == 'tsv':
		df = pd.read_csv(f, sep='\t').reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), sep='\t', index=False)
	elif fext == 'pkl':
		saved_model = pickle.load(f)
		with open(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), 'wb') as f:
			pickle.dump(saved_model, f)

		return render_template('load.html', k = 0, fname = fname, fext = fext, target_column = '')
	else:
		df = pd.DataFrame(['Not Available'], columns=['Col'])

	col_width = [(df[c].apply(lambda x: len(str(x))).values + [len(c)]).max() for c in df.columns]
	col_width = [cw * 7 + 10 for cw in col_width]
	if sum(col_width) > 1278:
		col_width = [0 if cw > 1278 / len(col_width) else cw for cw in col_width]

	col_width = [(1278 - sum(col_width)) / len([cw for cw in col_width if cw == 0]) if cw == 0 else cw for cw in col_width]

	json_data = collections.OrderedDict()
	json_data['rows'] = df.to_dict(orient='records')
	jsonp = json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', ''))

	return render_template('main.html', recent = [], fname = fname, fext = fext, column = df.columns.tolist(), colw = col_width, data = jsonp, flag='upload')

def save_model_outputs(outputs, fname, cluster_K):
	if 'model_output' not in os.listdir():
		os.mkdir('model_output')

	todaystr = datetime.today().strftime("%Y%m%d")
	existedResult = len([f for f in os.listdir('./model_output') if '%s_%d_%s' % (fname, cluster_K, todaystr) in f])
	resultfname = '%s_%d_%s.pkl' % (fname, cluster_K, todaystr) if existedResult == 0 else '%s_%d_%s_%d.pkl' % (fname, cluster_K, todaystr, (existedResult+1))

	with open(os.path.join(os.getcwd(), 'model_output', resultfname), 'wb') as f:
		pickle.dump(outputs, f)
	f.close()

@app.route('/upload_custom_entity', methods=['POST'])
def upload_custom_entity():
	fname = request.form['fname']
	fext = request.form['fext']
	target_column_name = request.form['target_column']
	cluster_K = request.form['k']

	return render_template('custom_entity.html', k = cluster_K, fname = fname, fext = fext, target_column = target_column_name)

@app.route('/prepare_model', methods=['POST'])
def prepare_model():
	entity_flag = request.form['entity_flag']
	global file_name
	file_name = request.form['fname']
	global file_ext
	file_ext = request.form['fext']

	target_column_name = request.form['target_column']
	cluster_K = request.form['k']

	ENTITY_HEADERS = ['entity_term', 'entity_type']
	if entity_flag == 'true':
		f = request.files['file']
		global entity_file_name
		entity_file_name = f.filename.split('.')[0]
		global entity_file_ext
		entity_file_ext = f.filename.split('.')[1]
		er_algo = request.form['er_algo']

		if entity_file_ext == 'csv':
			df = pd.read_csv(f).reset_index(drop=True)[ENTITY_HEADERS]
			df = df.append(pd.DataFrame([[er_algo, 'algorithm']], columns=ENTITY_HEADERS))
			df.to_csv(os.path.join(os.getcwd(), 'entity_files/', '%s.%s' % (entity_file_name, entity_file_ext)), index=False)
		elif entity_file_ext == 'tsv':
			df = pd.read_csv(f, sep='\t').reset_index(drop=True)[ENTITY_HEADERS]
			df = df.append(pd.DataFrame([[er_algo, 'algorithm']], columns=ENTITY_HEADERS))
			df.to_csv(os.path.join(os.getcwd(), 'entity_files/', '%s.%s' % (entity_file_name, entity_file_ext)), sep='\t', index=False)

	return render_template('load.html', k = cluster_K, fname = file_name, fext = file_ext, target_column = target_column_name)

@app.route('/run_model', methods=['POST'])
def run_model():
	fname = request.form['fname']
	fext = request.form['fext']
	target_column_name = request.form['target_column']
	cluster_K = int(request.form['k']) if len(request.form['k']) > 0 else 8

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json = run_topic_modeling(cluster_K = cluster_K, fname=fname, fext=fext, target_column_name=target_column_name)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json)
	save_model_outputs(outputs, fname, cluster_K)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json,
							km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/import_model', methods=['POST'])
def import_model():

	fname = request.form['fname']
	fext = request.form['fext']

	with open(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), 'rb') as f:
		saved_model = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json = load_topic_modeling(saved_model)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json)
	save_model_outputs(outputs, fname, len(lda_hbar_json['labels'])-1)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json,
							km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/load_model/<model_name>', methods=['GET', 'POST'])
def load_model(model_name):

	with open(os.path.join(os.getcwd(), 'model_output', '%s.pkl' % model_name), 'rb') as f:
		outputs = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json, metrics_json = outputs

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json, metrics_json = metrics_json)

@app.route('/detail', methods=['POST'])
def load_detail():
	barjson = json.loads(request.form['barjson'])
	doctable = json.loads(request.form['doctable'])
	type = json.loads(request.form['typejson'])
	dist = pd.DataFrame(doctable['rows']).groupby('topic_'+type)['document'].count().to_json()

	return render_template('detail.html', bar_json = barjson, document_table_json = doctable, distrib_json = dist, type_json = type)

@app.route('/entities', methods=['POST'])
def get_entities():
	import numpy as np
	# dic_df = pd.read_csv(os.path.join(os.getcwd(), 'entity_files/', '%s.%s' % (entity_file_name, entity_file_ext)))
	# print(dic_df.loc[dic_df.shape[0]-1])

	spacy.util.set_data_path('./py_source/nlp_data')
	nlp = spacy.load('en')

	type = 'topic_'+request.form['type']
	# print(type)
	docdata = json.loads(request.form['data'])
	df_rows = pd.DataFrame(docdata['rows'])
	# print(df_rows)
	df_rows['document'] = df_rows.document.apply(lambda x: ' '.join(str(x).split()))
	df_rows['length'] = df_rows.document.apply(lambda x: len(str(x)))

	max_len = 90000 if 90000 < df_rows.shape[0] else df_rows.shape[0]

	ent_list = []
	for ind, grp in df_rows.groupby(type):
		cum_len = 0
		start_ind = 0

		for idx, row in grp.reset_index().iterrows():
			cum_len += row.length
			if cum_len >= max_len :
				sub_text = grp.loc[start_ind:idx]['document']
				# print(sub_text)
				start_ind =idx

				# print(' '.join(sub_text.tolist()))
				doc = nlp(' '.join(sub_text.tolist()))

				for ent in doc.ents:
					if len(ent.text) >2:
						ent_list.append({'cluster':ind, 'text':ent.text, 'label':ent.label_})

	if len(ent_list) == 0:
		for ind in range(len(df_rows[type].unique())):
			ent_list.append({'cluster': ind, 'text': np.nan, 'label': np.nan})

	ent_df = pd.DataFrame(ent_list).groupby(['cluster', 'label']).count()
	print(ent_df)
	json_data = collections.OrderedDict()
	json_data['counts'] = ent_df.reset_index().to_dict(orient='records')
	return json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', '')

@app.route('/entity', methods=['POST'])
def get_entity():
	target_text = request.form['content']

	spacy.util.set_data_path('./py_source/nlp_data')
	nlp = spacy.load('en')
	doc = nlp(target_text)
	dochtml = displacy.render(doc, style='ent')

	return json.dumps({ 'dochtml': dochtml }, ensure_ascii=False, indent='\t')

# def init_db():
# 	db = getattr(g, '_database', None)
# 	if db is None:
# 		db = g._database = sqlite3.connect(os.path.join(os.getcwd(), 'db/data.db'))
# 	return db

if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

	# global db
	# db = init_db()
    print("end")
