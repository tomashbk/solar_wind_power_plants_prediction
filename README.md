Solar PV and Wind Power Plants Prediction
==============================
A web application to predict if a certain location in the world is convenient or not to install a Solar PV or Wind Power Plant according to the climatic factors of that location.

![Map Solar USA](/reports/figures/img_readme/usa_solar.jpg)

![Map Wind Arg](/reports/figures/img_readme/arg_wind.jpg)

The dataset used to train the model is the **Global Power Plant Database** from **World Resources Institute** (https://www.wri.org/) in combination of data from **NASA Prediction Of Wordlwide Energy Resources (POWER)** (https://power.larc.nasa.gov/).

The algorithm chosen for it had the best accuracy of all tried is a XGBoost Classifier.

![XGBoost Classification Report](/reports/figures/img_readme/class_report.jpg)

## Web Application
- `src/app/app.py`

## Data manipulation and model deploy
1. `notebooks/1-EDA-modif-GPPD.ipynb`
2. `src/data/2-make_dataset.py`
3. `notebooks/3-Analysis-and-modeling.ipynb`

## Model
- `models/xgbclass_model_1.pkl`


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
