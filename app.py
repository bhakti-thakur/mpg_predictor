from flask import Flask, jsonify, request
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins=["https://bhakti-thakur.github.io"])

model = joblib.load('mpg_model.pkl')

@app.route('/')
def home():
    return "MPG Predictor API Running!"

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'mpg': float(prediction[0])})

if __name__=='__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)