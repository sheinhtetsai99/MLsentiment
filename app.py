import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
import os

app = Flask(__name__)
model = load('model.joblib')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    text = [request.form['sentence']]
    print(text)
    result = model.predict(text)
    print(result)
    return render_template('index.html', result = result[0])

port = int(os.getenv("PORT",0))
if __name__ == '__main__':
    if port != 0:
        app.run(debug=True, host='0.0.0.0',port=port)
    else:
        app.run(debug=True)