from flask import Flask, jsonify;
from flask_cors import CORS;
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return ("sentiment analysis")

sentiment = 0.0

@app.route('/sentiment', methods=['GET'])
def returnSentiment():
    global sentiment
    if sentiment > 0:
        return ("positive")
    elif sentiment == 0:
        return ("neutral")
    else:
        return ("negative")

if __name__ == '__main__':
    app.run(debug=True)