from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
	return '<a href="/user/Kyle">Click</a>'

@app.route('/user/<name>')
def user(name):
	return '<h1>Hello, %s!</h1>' % name    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4040, debug=True)
