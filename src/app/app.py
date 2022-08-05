import pyhere
import sys
sys.path.insert(0, str(pyhere.here().resolve().joinpath("src")))
import utils

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, make_response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.form.get("type") == 'classification':
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))
        model = joblib.load(utils.DIR_MODELS/"xgbclass_model_1.pkl")
        df_classification_y_real_values = pd.read_csv(utils.DIR_DATA_PROCESSED/"classification_y_real_values.csv", index_col=0)
        results_lat_lon = utils.fetch_data_latitude_longitude_for_classification(latitude, longitude)
        int_prediction = int(model.predict(results_lat_lon))
        prediction = df_classification_y_real_values.squeeze().to_list()[int_prediction]
    elif request.form.get("type") == 'regression':
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))
        capacity_mw = int(request.form.get("capacity_mw"))
        # model = joblib.load(utils.DIR_MODELS/"knn_model_regressor.pkl")
        model = joblib.load(utils.DIR_MODELS/"rf_model_regressor.pkl")
        model_95 = joblib.load(utils.DIR_MODELS/"qr_model_95.pkl")
        model_05 = joblib.load(utils.DIR_MODELS/"qr_model_05.pkl")
        # df_classification_y_real_values = pd.read_csv(utils.DIR_DATA_PROCESSED/"classification_y_real_values.csv", index_col=0)
        results_lat_lon = utils.fetch_data_latitude_longitude_for_regression(latitude, longitude, capacity_mw)
        # int_prediction = int(model.predict(results_lat_lon))
        # prediction = df_classification_y_real_values.squeeze().to_list()[int_prediction]
        ndarray_prediction = model.predict(results_lat_lon)
        ndarray_max_pred = model_95.predict([ndarray_prediction])
        ndarray_min_pred = model_05.predict([ndarray_prediction])
        prediction = str(round(ndarray_prediction[0],2)) + " GWh (Min: " + str(round(ndarray_min_pred[0],2)) + " GWh, Max: " + str(round(ndarray_max_pred[0],2)) + " GWh)"
    
    return prediction
    

if __name__ == "__main__":
    app.run(port=8080)

