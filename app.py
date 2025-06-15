from flask import Flask, jsonify, request
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('mpg_model.pkl')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'mpg': float(prediction[0])})

if __name__=='__main__':
    app.run(debug=True)