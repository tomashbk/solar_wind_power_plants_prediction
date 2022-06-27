import pyhere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

import certifi
import urllib3
import os, json, requests
from io import StringIO

# GLOBAL VARIABLES
DIR_DATA_RAW = pyhere.here().resolve().joinpath("data", "raw")
DIR_DATA_INTERIM = pyhere.here().resolve().joinpath("data", "interim")
DIR_DATA_EXTERNAL = pyhere.here().resolve().joinpath("data", "external")
DIR_DATA_PROCESSED = pyhere.here().resolve().joinpath("data", "processed")
DIR_MODELS = pyhere.here().resolve().joinpath("models")
SEASONS = {'autumn', 'spring', 'summer', 'winter'}
YEARS = {'2013', '2014', '2015', '2016', '2017', '2018'}
NORTH_HEMISPHERE_MONTHS_SEASONS = {
                                    "autumn": np.array(["OCT", "NOV", "DEC"]),
                                    "winter": np.array(["JAN", "FEB", "MAR"]),
                                    "spring": np.array(["APR", "MAY", "JUN"]),
                                    "summer": np.array(["JUL", "AUG", "SEP"])
                                    }
SOUTH_HEMISPHERE_MONTHS_SEASONS = {
                                    "autumn": np.array(["APR", "MAY", "JUN"]),
                                    "winter": np.array(["JUL", "AUG", "SEP"]),
                                    "spring": np.array(["OCT", "NOV", "DEC"]),
                                    "summer": np.array(["JAN", "FEB", "MAR"])
                                    }
MONTHS_OF_YEAR = np.array(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])
# NORTH_HEMISPHERE_MONTHS_SEASONS["autumn"] = np.array(["OCT", "NOV", "DEC"])
# NORTH_HEMISPHERE_MONTHS_SEASONS["winter"] = np.array(["JAN", "FEB", "MAR"])
# NORTH_HEMISPHERE_MONTHS_SEASONS["spring"] = np.array(["APR", "MAY", "JUN"])
# NORTH_HEMISPHERE_MONTHS_SEASONS["summer"] = np.array(["JUL", "AUG", "SEP"])
# SOUTH_HEMISPHERE_MONTHS_SEASONS["spring"] = np.array(["OCT", "NOV", "DEC"])
# SOUTH_HEMISPHERE_MONTHS_SEASONS["summer"] = np.array(["JAN", "FEB", "MAR"])
# SOUTH_HEMISPHERE_MONTHS_SEASONS["autumn"] = np.array(["APR", "MAY", "JUN"])
# SOUTH_HEMISPHERE_MONTHS_SEASONS["winter"] = np.array(["JUL", "AUG", "SEP"])


def make_mi_scores(X, y):
    """
    Function to calculate Mutual Information scores.
    """
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def get_accuracy(max_leaf_nodes, X_train, X_test, y_train, y_test):
    """
    Function to calculate the Accuracy of different 
    Decision Tree Classifiers given a number of max_leaf_nodes.
    """
    model = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_val)

    return accuracy

def get_accuracy_knn(n_neighbors, X_train, X_test, y_train, y_test):
    """
    Function to calculate the Accuracy of different 
    Decision Tree Classifiers given a number of max_leaf_nodes.
    """
    model = KNeighborsClassifier(n_neighbors = n_neighbors)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_val)

    return accuracy

def plot_mi_scores(scores):
    """
    Function to plot Mutual Information scores.
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def correlation_matrix(df: pd.DataFrame):
    """
    Function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # CREATE THE MATRIX
    matrix = df.corr()
    
    # CREATE CMAP
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)
    # CREATE A MASK
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # MAKE FIGSIZE BIGGER
    fig, ax = plt.subplots(figsize=(16,12))
    
    # PLOT THE MATRIX
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap, ax=ax)

def balancing_data_more_than_1000(dataframe, target_column):
    """
    Function to balance data if the target with less samples is more than 1000.
    """
    target_with_less_value = dataframe[[target_column]].value_counts().sort_values().index[0][0]
    value_of_target_with_less_value = dataframe[[target_column]].value_counts().sort_values()[0]
    dict_outbalanced_targets_values = dataframe[[target_column]].value_counts().sort_values()[1:].to_dict()

    dict_outbalanced_targets_values
    if(value_of_target_with_less_value > 1000):
        for key_tuple, value in dict_outbalanced_targets_values.items():
            difference = (value - value_of_target_with_less_value)
            index_rows_to_delete = dataframe[dataframe[target_column] == key_tuple[0]].sample(difference).index
            dataframe.drop(index_rows_to_delete, axis = 0, inplace = True)

def custom_classification_prediction_report(model, X, y, X_test, y_test, list_target_in_order):
    """
    Function to show a custom report of a classification prediction.
    It includes the accuracy score, a cross validation score, the classification_report from sklearn, and a Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    print(f'{np.around(model.score(X_test, y_test) * 100, 2)}%')
    # If data is unordered in nature (i.e. non - Time series) then shuffle = True is right choice.
    results_cvs = cross_val_score(model, X, y, cv=StratifiedKFold(shuffle = True))
    print(f'{np.around(results_cvs * 100, 2)}(%)')
    print(f'Mean: {np.around(results_cvs.mean() * 100, 2)}%, Standard Deviation: {np.around(results_cvs.std() * 100, 2)}%')
    print(classification_report(y_test, y_pred, target_names=list_target_in_order))
    confusion_matrix_return = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix_return, annot=True, fmt = 'g')

def calculate_feature_mean_std(X):
    climate_features = {"ALLSKY_SFC_SW_DWN",
                        "CLRSKY_SFC_SW_DWN",
                        # "ALLSKY_SFC_SW_DIFF",
                        # "ALLSKY_SFC_SW_UP",
                        "ALLSKY_SFC_LW_DWN",
                        "ALLSKY_SFC_LW_UP",
                        "ALLSKY_SFC_SW_DNI",
                        # "ALLSKY_SFC_SW_DNI_MAX_RD",
                        # "ALLSKY_SFC_SW_UP_MAX",
                        # "CLRSKY_SFC_SW_DIFF",
                        "CLRSKY_SFC_SW_DNI",
                        # "CLRSKY_SFC_SW_UP",
                        #"ALLSKY_KT",
                        "WS10M_MAX_AVG",
                        "WS50M_MAX_AVG",
                        "WS50M",
                        # "WS50M_RANGE_AVG",
                        "WS10M",
                        "T2M",
                        # "WS10M_RANGE_AVG"
                    }   

    list_total = []
    for feature in climate_features:
        feature_string = ""
        
        for season in SEASONS:
            dict_features_to_apply_mean = {}
            list_to_append = []
            for year in YEARS:
                feature_string = f"{season}_{feature}_{year}"
                
                list_to_append.append(feature_string)
            dict_features_to_apply_mean = {f"{season}_{feature}": list_to_append}
        
            list_total.append(dict_features_to_apply_mean)
    for dict_season_feature in list_total:
        for season_feature in dict_season_feature:
            X[f'mean_{season_feature}']= X[dict_season_feature[season_feature]].mean(axis=1)
            X[f'std_{season_feature}']= X[dict_season_feature[season_feature]].std(axis=1)
    


def fetch_data_latitude_longitude(latitude, longitude):

    
    url_parameters = ["ALLSKY_SFC_SW_DWN",
                        "CLRSKY_SFC_SW_DWN",
                        "ALLSKY_SFC_SW_DIFF",
                        "ALLSKY_SFC_SW_UP",
                        "ALLSKY_SFC_LW_DWN",
                        "ALLSKY_SFC_LW_UP",
                        "ALLSKY_SFC_SW_DNI",
                        # "ALLSKY_SFC_SW_DNI_MAX_RD",
                        "ALLSKY_SFC_SW_UP_MAX",
                        "CLRSKY_SFC_SW_DIFF",
                        "CLRSKY_SFC_SW_DNI",
                        "CLRSKY_SFC_SW_UP",
                        #"ALLSKY_KT",
                        "WS10M_MAX_AVG",
                        "WS50M_MAX_AVG",
                        "WS50M",
                        "WS50M_RANGE_AVG",
                        "WS10M",
                        "WS10M_RANGE_AVG",
                        "T2M"
                    ]
    # columns_to_drop = [
    #                     'capacity_mw',
    #                     'latitude',
    #                     'longitude',
    #                     'primary_fuel_transformed',
    #                     'generation_gwh_2013',
    #                     'generation_gwh_2014',
    #                     'generation_gwh_2015',
    #                     'generation_gwh_2016',
    #                     'generation_gwh_2017',
    #                     'generation_gwh_2018',
    #                     'generation_gwh_2019'
    #                 ]

    base_url = r"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={url_parameters}&community=RE&longitude={longitude}&latitude={latitude}&start=2013&end=2019&format=CSV&header=false"
    df_response = pd.DataFrame()

    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )

    api_request_url = base_url.format(longitude=longitude, latitude=latitude, url_parameters=','.join(url_parameters))
    response = http.request('GET', api_request_url, timeout=30.00).data.decode('utf-8')
    df_response_aux = pd.read_csv(StringIO(response))
    df_response_aux["latitude"] = latitude
    df_response_aux["longitude"] = longitude
    # print(df_response_aux)
    if longitude > 0:
        hemisphere_months_seasons = NORTH_HEMISPHERE_MONTHS_SEASONS
    else:
        hemisphere_months_seasons = SOUTH_HEMISPHERE_MONTHS_SEASONS
    for index, element in hemisphere_months_seasons.items():
        df_response_aux[index]= df_response_aux[element].mean(axis=1)

    df_response_aux.drop(columns= MONTHS_OF_YEAR, inplace = True)

    # "PIVAT! PIVAT! PIVAT!"
    df_response_aux = df_response_aux.pivot_table(index=["latitude", "longitude"], columns=["PARAMETER", "YEAR"])
    df_response_aux.columns = ["_".join(map(str, cols)) for cols in df_response_aux.columns.to_flat_index()]

    if(df_response.empty):
    
        df_response = df_response_aux.copy()
    else:
        df_response = pd.concat([df_response,df_response_aux])



    df_response.reset_index(inplace = True)
    
    
    calculate_feature_mean_std(df_response)

    columns_delete = df_response.columns.str.contains('latitude') | df_response.columns.str.contains('longitude') | df_response.columns.str.contains('2019') | df_response.columns.str.contains('2012') | df_response.columns.str.contains('2013') | df_response.columns.str.contains('2014') | df_response.columns.str.contains('2015') | df_response.columns.str.contains('2016') | df_response.columns.str.contains('2017') | df_response.columns.str.contains('2018') |  df_response.columns.str.contains('ANN') |  df_response.columns.str.contains('LW') |  df_response.columns.str.contains('WS10') | df_response.columns.str.contains('MAX')                  
    
    df_response = df_response.loc[:,~columns_delete]
    df_response = df_response.reindex(sorted(df_response.columns), axis=1)
    return df_response

