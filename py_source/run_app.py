from flask import Flask, request, render_template

app = Flask(__name__, template_folder='../web/', static_folder='../web/resource')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/user/<name>', methods=['GET', 'POST'])
def user(name):
	return render_template('user.html', name=name)

if __name__ == '__main__':
    print("start")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    print("end")
