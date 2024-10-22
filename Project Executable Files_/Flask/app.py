import numpy as np
import pandas as pd
import pickle
import joblib
import time
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Load the model and scaler (assuming they're in the correct locations)
model = pickle.load(open('xgbmodel2.pkl', 'rb'))
scale = pickle.load(open("scaler_model2.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']
    value = pd.DataFrame(features_values, columns=names)
    value = scale.transform(value)

    prediction = model.predict(value)
    if prediction[0] == 0:
        prediction1 = "Not a potential customer"
        return render_template("nopotential.html", predict=prediction1)
    elif prediction[0] == 1:
        prediction1 = "Potential customer"
        return render_template("potential.html", predict=prediction1)
    else:
        prediction1 = "Highly potential customer"

    return render_template("highlypot.html", predict=prediction1)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=True)