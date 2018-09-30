# -- coding: utf-8 --

from flask import Flask, request, render_template, jsonify, escape
import pandas as pd
import json
import collections

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
	return render_template('grid.html', fext='none', column = [], data = '{}')

@app.route('/upload_file', methods=['POST'])
def upload_file():
	f = request.files['file']
	fext = f.filename.split('.')[1]
	if fext == 'csv':
		df = pd.read_csv(f, index_col=0).reset_index(drop=True)
	elif fext == 'tsv':
		df = pd.read_csv(f, sep='\t', index_col=0).reset_index(drop=True)
	else:
		df = pd.DataFrame(['Not Available'], columns=['Col'])

	json_data = collections.OrderedDict()
	json_data['rows'] = df.to_dict(orient='records')
	jsonp = json.loads(json.dumps(json_data, ensure_ascii=False, indent='\t').replace('`', ''))

	return render_template('grid.html', fext = fext, column = df.columns.tolist(), data = jsonp)


if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
