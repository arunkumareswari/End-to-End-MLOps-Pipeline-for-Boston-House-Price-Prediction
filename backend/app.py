from flask import Flask, request, jsonify, url_for, render_template
import os
import json
import pickle
import numpy as np
import pandas as pd 

app = Flask(__name__, template_folder = "../frontend")

# Get the directory where app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to this directory
model_path = os.path.join(base_dir, "..", "model", "model.pkl")
scaler_path = os.path.join(base_dir, "..", "model", "scaler.pkl")

reg_model = pickle.load(open(model_path, 'rb'))
scalar = pickle.load(open(scaler_path, 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_api", methods= ["POST"])
def predict_api():
    # get data from user
    data = request.json['data']
    print(data)

    # convert input values to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)

    #scale input values
    scaled_input = scalar.transform(input_array)

    #predict 
    prediction = reg_model.predict(scaled_input)

    #return prediction
    return jsonify({"prediction": round(float(prediction[0]),2)})


# FROM PREDICTION

@app.route("/predict", methods=["POST"])
def predict():
    # get data from user
    data = [float(x) for x in request.form.values()]

    # convert input values to numpy array and scale
    input_array = np.array(data).reshape(1, -1)
    scaled_input = scalar.transform(input_array)

    #predict 
    prediction = reg_model.predict(scaled_input)
    return render_template(
        "index.html",
        prediction_text = f"The House price prediction is {prediction[0]:.2f}."
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)

