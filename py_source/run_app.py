from flask import Flask, request, render_template
import pandas as pd

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
	return render_template('grid.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
	f = request.files['file']
	fext = f.filename.split('.')[1]
	if fext == 'csv':
		df = pd.read_csv(f)
		table_html = df.to_html(index=False)
	elif fext == 'tsv':
		df = pd.read_csv(f, sep='\t')
		table_html = df.to_html(index=False)
	else:
		table_html = 'This file is not available format!! Please upload csv or tsv only'

	return render_template('grid.html', df = table_html.replace('`', '\''))


if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
