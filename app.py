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
    data = request.json
    text = data['data']
    result =  model.predict([text])
    return_dict = {}
    return_dict['result'] = result[0]
    print(return_dict)
    return return_dict

port = int(os.getenv("PORT",0))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080)