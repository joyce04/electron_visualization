from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
	# return '<a href="/user/Kyle">Click</a>'
	return '<h1>Hello World</h1>'

@app.route('/user/<name>')
def user(name):
	return '<h1>Hello, %s!</h1>' % name

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
