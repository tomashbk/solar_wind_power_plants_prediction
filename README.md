Solar PV and Wind Power Plants Prediction
==============================

A Data Science Project to predict if a certain location in the World is convenient or not to install a Solar PV or Wind Power Plant according to climatic factors of that location, and to predict its anual generation with 90% of confidence.

This project was presented for the Data Science Diploma course at **Instituto Data Science** (https://institutodatascience.org/)

This project has the code to manipulate the data, building models and to deploy in a web application.

<!-- ![Map Solar USA](/reports/figures/img_readme/usa_solar.jpg) -->

<!-- ![Map Wind Arg](/reports/figures/img_readme/arg_wind.jpg) -->

The dataset used to train the model is the **Global Power Plant Database** from **World Resources Institute** (https://www.wri.org/) in combination of data from **NASA Prediction Of Worldwide Energy Resources (POWER)** (https://power.larc.nasa.gov/).


<!-- ![XGBoost Classification Report](/reports/figures/img_readme/class_report.jpg) -->

## Web Application
- `src/app/app.py`

## Data manipulation and model deploy
1. `notebooks/1-EDA-modif-GPPD.ipynb`
2. `src/data/2-make_dataset.py`
3. `notebooks/3-classification.ipynb`
4. `notebooks/4-regressions_generation_solar.ipynb`
5. `notebooks/5-regressions_generation_wind.ipynb`

## Models
- `models/rf_model2.pkl`
- `models/solar_rf_model_regressor.pkl`
- `models/wind_rf_model_regressor.pkl`
- `models/solar_qr_model_05.pkl`
- `models/solar_qr_model_95.pkl`
- `models/wind_qr_model_05.pkl`
- `models/wind_qr_model_95.pkl`


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
