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
    prediction = {}
    if request.form.get("type") == 'classification':
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))
        model = joblib.load(utils.DIR_MODELS/"rf_model2.pkl")
        df_classification_y_real_values = pd.read_csv(utils.DIR_DATA_PROCESSED/"classification_y_real_values.csv", index_col=0)
        results_lat_lon = utils.fetch_data_latitude_longitude_for_classification(latitude, longitude)
        int_prediction = int(model.predict(results_lat_lon))
        prediction_result = df_classification_y_real_values.squeeze().to_list()[int_prediction]
        if prediction_result == 'Wind':
            prediction['capacity_min_max']  = pd.read_csv(utils.DIR_DATA_PROCESSED/'wind_minmax_capacity.csv', index_col=0).to_json(orient='records')
        if prediction_result == 'Solar':
            prediction['capacity_min_max']  = pd.read_csv(utils.DIR_DATA_PROCESSED/'solar_minmax_capacity.csv', index_col=0).to_json(orient='records')
        
        prediction['result'] = prediction_result
    elif request.form.get("type") == 'regression':
        latitude = float(request.form.get("latitude"))
        longitude = float(request.form.get("longitude"))
        capacity_mw = int(request.form.get("capacity_mw"))
        category = request.form.get("category")
        if category == 'wind':
            model = joblib.load(utils.DIR_MODELS/"wind_rf_model_regressor.pkl")
            model_95 = joblib.load(utils.DIR_MODELS/"wind_qr_model_95.pkl")
            model_05 = joblib.load(utils.DIR_MODELS/"wind_qr_model_05.pkl")
        if category == 'solar':
            model = joblib.load(utils.DIR_MODELS/"solar_rf_model_regressor.pkl")
            model_95 = joblib.load(utils.DIR_MODELS/"solar_qr_model_95.pkl")
            model_05 = joblib.load(utils.DIR_MODELS/"solar_qr_model_05.pkl")
        results_lat_lon = utils.fetch_data_latitude_longitude_for_regression(latitude, longitude, capacity_mw, category)
        ndarray_prediction = model.predict(results_lat_lon)
        ndarray_max_pred = model_95.predict([ndarray_prediction])
        ndarray_min_pred = model_05.predict([ndarray_prediction])
        prediction_result = str(round(ndarray_prediction[0],2)) + " GWh (Min: " + str(round(ndarray_min_pred[0],2)) + " GWh, Max: " + str(round(ndarray_max_pred[0],2)) + " GWh)"
        prediction['result'] = prediction_result

    return prediction
    

if __name__ == "__main__":
    app.run(port=8080)

