# -- coding: utf-8 --

from flask import Flask, request, render_template, jsonify, escape
import pandas as pd
import json
import collections
import os
import pickle
import spacy

from spacy import displacy
from datetime import datetime
from werkzeug.utils import secure_filename
from topic_modeling import run_topic_modeling, load_topic_modeling

app = Flask(__name__, template_folder='./web/', static_folder='./web')

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
		df = pd.read_csv(f, index_col=0).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)))
	elif fext == 'tsv':
		df = pd.read_csv(f, sep='\t', index_col=0).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), sep='\t')
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

@app.route('/prepare_model', methods=['POST'])
def prepare_model():
	fname = request.form['fname']
	fext = request.form['fext']
	target_column_name = request.form['target_column']
	cluster_K = request.form['k']

	return render_template('load.html', k = cluster_K, fname = fname, fext = fext, target_column = target_column_name)

@app.route('/run_model', methods=['POST'])
def run_model():
	fname = request.form['fname']
	fext = request.form['fext']
	target_column_name = request.form['target_column']
	cluster_K = int(request.form['k']) if len(request.form['k']) > 0 else 8

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json = run_topic_modeling(cluster_K = cluster_K, fname=fname, fext=fext, target_column_name=target_column_name)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json)
	save_model_outputs(outputs, fname, cluster_K)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json)

@app.route('/import_model', methods=['POST'])
def import_model():

	fname = request.form['fname']
	fext = request.form['fext']

	with open(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), 'rb') as f:
		saved_model = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json = load_topic_modeling(saved_model)

	outputs = (lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json)
	save_model_outputs(outputs, fname, len(lda_hbar_json['labels'])-1)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json)

@app.route('/load_model/<model_name>', methods=['GET', 'POST'])
def load_model(model_name):

	with open(os.path.join(os.getcwd(), 'model_output', '%s.pkl' % model_name), 'rb') as f:
		outputs = pickle.load(f)
	f.close()

	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json = outputs

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json)

@app.route('/detail', methods=['POST'])
def load_detail():
	barjson = json.loads(request.form['barjson'])
	doctable = json.loads(request.form['doctable'])
	type = json.loads(request.form['typejson'])
	dist = pd.DataFrame(doctable['rows']).groupby('topic_'+type)['document'].count().to_json()

	return render_template('detail.html', bar_json = barjson, document_table_json = doctable, distrib_json = dist, type_json = type)

@app.route('/entities', methods=['POST'])
def get_entities():
	spacy.util.set_data_path('./py_source/nlp_data')
	nlp = spacy.load('en')
	max_len = 90000

	type = 'topic_'+request.form['type']
	docdata = json.loads(request.form['data'])
	df_rows = pd.DataFrame(docdata['rows'])
	df_rows['document'] = df_rows.document.apply(lambda x: ' '.join(str(x).split()))
	df_rows['length'] = df_rows.document.apply(lambda x: len(str(x)))

	ent_list = []
	for ind, grp in df_rows.groupby(type):
		cum_len = 0
		start_ind = 0

		for idx, row in grp.reset_index().iterrows():
			cum_len += row.length
			if cum_len >= max_len :
				sub_text = grp.iloc[start_ind:idx]['document']
				start_ind =idx
				cum_len = 0
				# print(len(' '.join(sub_text.tolist())))
				doc = nlp(' '.join(sub_text.tolist()))

				for ent in doc.ents:
					if len(ent.text) >2:
						ent_list.append({'cluster':ind, 'text':ent.text, 'label':ent.label_})

	ent_df = pd.DataFrame(ent_list).groupby(['cluster', 'label']).count()
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

if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
