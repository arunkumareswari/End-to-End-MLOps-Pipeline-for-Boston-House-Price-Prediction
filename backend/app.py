from flask import Flask, request, jsonify, url_for, render_template

import json
import pickle
import numpy as np
import pandas as pd 

app = Flask(__name__, template_folder = "../frontend")

# load model

reg_model = pickle.load(open('../model/model.pkl', 'rb'))
scalar = pickle.load(open('../model/scaler.pkl', 'rb'))

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
    return jsonify({"prediction": prediction[0]})


# FROM PREDICTION

@app.route("/predict", methods=["POST"])
def predict():
    # get data from user
    data = [float(X) for x in request.form.values()]

    # convert input values to numpy array and scale
    input_array = np.array(data).reshape(1, -1)
    scaled_input = scalar.transform(input_array)

    #predict 
    prediction = reg_model.predict(scaled_input)
    return render_template(
        "index.html",
        prediction_text = f"The House price prediction is {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)

