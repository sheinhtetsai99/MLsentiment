import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
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

if __name__ == '__main__':
    app.run(debug=True)