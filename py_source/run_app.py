# -- coding: utf-8 --

from flask import Flask, request, render_template, jsonify, escape
import pandas as pd
import json
import collections
import os
from werkzeug.utils import secure_filename
from topic_modeling_three_models import run_topic_modeling

app = Flask(__name__, template_folder='../web/', static_folder='../web')

@app.route('/')
def index():
	"""
	Rerount to index html
	"""
	return render_template('index.html')

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
	f = request.files['file']
	fname = f.filename.split('.')[0]
	fext = f.filename.split('.')[1]
	if fext == 'csv':
		df = pd.read_csv(f, index_col=0).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)))
	elif fext == 'tsv':
		df = pd.read_csv(f, sep='\t', index_col=0).reset_index(drop=True)
		df.to_csv(os.path.join(os.getcwd(), '%s.%s' % (fname, fext)), sep='\t')
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

	return render_template('grid.html', fname = fname, fext = fext, column = df.columns.tolist(), colw = col_width, data = jsonp, flag='upload')

@app.route('/run_model', methods=['POST'])
def run_model():
	fname = request.form['fname']
	fext = request.form['fext']
	target_column_name = request.form['target_column']
	
	lda_hbar_json, km_hbar_json, dec_hbar_json, lda_scatter_json, km_scatter_json, dec_scatter_json, document_table_json = run_topic_modeling(fname=fname, fext=fext, target_column_name=target_column_name, train_flag=False)

	return render_template('visual.html', lda_hbar_json = lda_hbar_json, km_hbar_json = km_hbar_json, dec_hbar_json = dec_hbar_json, lda_scatter_json = lda_scatter_json, km_scatter_json = km_scatter_json, dec_scatter_json = dec_scatter_json, document_table_json = document_table_json)

if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
