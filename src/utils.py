import pyhere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, validation_curve, learning_curve, cross_validate

import certifi
import urllib3
import os, json, requests
from io import StringIO
import datetime

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


def make_mi_scores(X, y, type):
    """
    Function to calculate Mutual Information scores.
    """
    if type == "classifier":
        mi_scores = mutual_info_classif(X, y)
    if type == "regression":
        mi_scores = mutual_info_regression(X, y)

    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def get_accuracy_tree(type, max_leaf_nodes, X_train, X_test, y_train, y_test):
    """
    Function to calculate the Accuracy of different 
    Decision Tree given a number of max_leaf_nodes.
    """
    if type == "classifier":
        model = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state=0)
        model.fit(X_train, y_train)
        # preds_val = model.predict(X_test)
        # score = accuracy_score(y_test, preds_val)
        score = model.score(X_test, y_test)
    if type == "regression":
        model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    

    return score

def get_accuracy_knn(type, n_neighbors, X_train, X_test, y_train, y_test):
    """
    Function to calculate the Accuracy of different 
    KNN given a number of n_neighbors.
    """
    if type == "classifier":
        model = KNeighborsClassifier(n_neighbors = n_neighbors)
        model.fit(X_train, y_train)
        preds_val = model.predict(X_test)
        score = accuracy_score(y_test, preds_val)
    if type == "regression":
        model = KNeighborsRegressor(n_neighbors = n_neighbors)
        model.fit(X_train, y_train)
        # preds_val = model.predict(X_test)
        score = model.score(X_test, y_test)

    

    return score

def plot_scores(scores, title):
    """
    Function to plot Scores.
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title(title)


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
    # If data is unordered in nature (i.e. non - Time series) then shuffle = True is right choice.
    results_cvs = cross_val_score(model, X, y, cv=StratifiedKFold(shuffle = True))
    print('Cross validation:')
    print(f'{np.around(results_cvs * 100, 2)}(%)')
    print(f'Mean: {np.around(results_cvs.mean() * 100, 2)}%, Standard Deviation: {np.around(results_cvs.std() * 100, 2)}%')
    
    print('Hold Out:')
    y_pred = model.predict(X_test)
    print(f'{np.around(model.score(X_test, y_test) * 100, 2)}%')
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
    


def fetch_data_latitude_longitude_for_classification(latitude, longitude):

    
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


def fetch_data_latitude_longitude_for_regression(latitude, longitude, capacity_mw):
    year_to_fetch_data = datetime.datetime.now().year-1
    hours_in_a_year = 365*24

    # url_parameters = ["ALLSKY_SFC_SW_DWN",
    #                     "CLRSKY_SFC_SW_DWN",
    #                 ] 
    url_parameters = [  
                        'ALLSKY_SFC_LW_UP',
                        'T2M',
                        'ALLSKY_SFC_SW_UP',
                        'ALLSKY_SFC_SW_DWN',
                        'CLRSKY_SFC_SW_UP',
                        'CLRSKY_SFC_SW_DWN',
                        'ALLSKY_SFC_SW_DNI',
                        'CLRSKY_SFC_SW_DNI',
                        'ALLSKY_SFC_SW_UP_MAX',
                        'ALLSKY_SFC_SW_DIFF',
                        'CLRSKY_SFC_SW_DIFF',
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

    base_url = r"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={url_parameters}&community=RE&longitude={longitude}&latitude={latitude}&start={year_to_fetch_data}&end={year_to_fetch_data}&format=CSV&header=false"
    df_response = pd.DataFrame()

    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )

    api_request_url = base_url.format(longitude=longitude, latitude=latitude, url_parameters=','.join(url_parameters), year_to_fetch_data=year_to_fetch_data)
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
    
    
    # calculate_feature_mean_std(df_response)
    dict_columns_rename = {a:a.replace(f'_{year_to_fetch_data}', '') for a in df_response.columns}
    df_response.rename(columns=dict_columns_rename, inplace=True)
    
    df_response['capacity_mw'] = capacity_mw

    # columns_delete = df_response.columns.str.contains('latitude') | df_response.columns.str.contains('longitude')
    
    columns_keep = [
                        'capacity_mw', 
                        'autumn_ALLSKY_SFC_LW_UP',
                        'winter_ALLSKY_SFC_LW_UP', 
                        'winter_T2M', 
                        'autumn_T2M',
                        'spring_ALLSKY_SFC_SW_UP', 
                        'ANN_ALLSKY_SFC_LW_UP',
                        'ANN_ALLSKY_SFC_SW_DWN', 
                        'winter_ALLSKY_SFC_SW_UP',
                        'autumn_ALLSKY_SFC_SW_DWN', 
                        'spring_CLRSKY_SFC_SW_UP', 
                        'ANN_T2M',
                        'autumn_ALLSKY_SFC_SW_UP', 
                        'ANN_ALLSKY_SFC_SW_UP',
                        'ANN_CLRSKY_SFC_SW_DWN', 
                        'spring_ALLSKY_SFC_SW_DWN',
                        'ANN_ALLSKY_SFC_SW_DNI', 
                        'autumn_ALLSKY_SFC_SW_DNI',
                        'spring_CLRSKY_SFC_SW_DNI', 
                        'spring_CLRSKY_SFC_SW_DWN',
                        'summer_CLRSKY_SFC_SW_DWN', 
                        'spring_ALLSKY_SFC_SW_DNI',
                        'winter_CLRSKY_SFC_SW_UP', 
                        'winter_ALLSKY_SFC_SW_DWN',
                        'summer_ALLSKY_SFC_SW_DWN', 
                        'summer_T2M', 
                        'winter_CLRSKY_SFC_SW_DWN',
                        'summer_ALLSKY_SFC_LW_UP', 
                        'spring_ALLSKY_SFC_SW_UP_MAX',
                        'summer_ALLSKY_SFC_SW_DNI', 
                        'summer_CLRSKY_SFC_SW_DNI',
                        'ANN_CLRSKY_SFC_SW_UP', 
                        'winter_ALLSKY_SFC_SW_UP_MAX',
                        'autumn_CLRSKY_SFC_SW_UP', 
                        'winter_ALLSKY_SFC_SW_DNI',
                        'ANN_CLRSKY_SFC_SW_DNI', 
                        'autumn_ALLSKY_SFC_SW_DIFF',
                        'autumn_CLRSKY_SFC_SW_DIFF', 
                        'autumn_CLRSKY_SFC_SW_DWN', 
                        'spring_T2M',
                        'spring_ALLSKY_SFC_LW_UP', 
                        'winter_ALLSKY_SFC_SW_DIFF',
                        'autumn_ALLSKY_SFC_SW_UP_MAX'
                    ]

    # df_response = df_response.loc[:,~columns_delete]
    df_response = df_response.loc[:,columns_keep]
    df_response = df_response.reindex(sorted(df_response.columns), axis=1)
    return df_response

def validation_curve_plot(plot_title, model, X, y, param_name, param_range):
    train_scores, test_scores = validation_curve(
            model,
            X,
            y,
            param_name = param_name,
            param_range = param_range,
            cv = 5
        )
        
    np.mean(train_scores, axis=1)
    
    plt.figure(figsize=(15,5))
    plt.plot(np.mean(train_scores, axis=1),
        label = "Training Score", color = 'b')
    plt.plot(np.mean(test_scores, axis=1),
    label = "Cross Validation Score", color = 'g')
    plt.xticks(np.arange(param_range.shape[0]), param_range)
    plt.title(plot_title)
    plt.legend()

def learning_curve_plot(plot_title, model_with_hp, X, y):
    lc = learning_curve(model_with_hp, X, y, cv=5)
    samples, train, test = lc[0], lc[1], lc[2]
    
    plt.figure(figsize=(15,5))
    plt.plot(samples, np.mean(train, axis=1),
        label = "Training Score", color = 'b')
    plt.plot(samples, np.mean(test, axis=1),
    label = "Cross Validation Score", color = 'g')
    plt.title(plot_title)
    plt.legend()


# def performance_metrics(y_true, y_pred, dataset_type):
def performance_metrics_cross_val(X, y, model_with_hp, dataset_type):
    results = cross_validate(
        model_with_hp,
        X, 
        y, 
        cv=5, 
        scoring=(
            'r2', 
            'neg_mean_squared_error', 
            'neg_mean_absolute_error',
            'neg_root_mean_squared_error')
    )

    dict_results = {}
    for i, val in results.items():
        if 'test' in i:
            if 'neg' in i:
                dict_results[i.replace("neg_", "")] = -np.mean(val)

                # if 'mean_squared_error' in i:
                #     dict_results['test_root_mean_squared_error'] = np.sqrt(-np.mean(val))
            else:
                dict_results[i] = np.mean(val)
    return dict_results
    # cross_val_score(lasso, X, y, cv=5)
    # cross_val_score(nb_model_1, X, y, cv=StratifiedKFold(shuffle = True))
    # r2 = r2_score(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = np.sqrt(mse) 
    # mae = mean_absolute_error(y_true, y_pred)
    # return pd.DataFrame({'metrica': ['R2', 'MSE', 'RMSE', 'MAE'],
    #                      'valor':[r2, mse, rmse, mae],
    #                      'dataset_type':dataset_type})
                         
    
def coef_summary(results):
    '''
    Toma los resultado del modelo de OLS 
    
    Elimina el intercepto.
    '''
    # Creo un dataframe de los resultados del summary 
    coef_df = pd.DataFrame(results.summary().tables[1].data)
    
    # Agrego el nombre de las columnas
    coef_df.columns = coef_df.iloc[0]

    # Elimino la fila extra del intercepto
    coef_df=coef_df.drop(0)

    # Seteo el nombre de las variables como index
    coef_df = coef_df.set_index(coef_df.columns[0])

    # Convertimos a float los object 
    coef_df = coef_df.astype(float)

    # Obtenemos el error; (coef - limite inferior del IC)
    errors = coef_df['coef'] - coef_df['[0.025']
    
    # Agregamos los errores al dataframe
    coef_df['errors'] = errors

    # Eliminamos la variable const
    coef_df = coef_df.drop(['const'])

    # Ordenamos los coeficientes 
    coef_df = coef_df.sort_values(by=['coef'])

    ### Graficamos ###

    # x-labels
    variables = list(coef_df.index.values)
    
    # Agregamos la columna con el nombre de las variables
    coef_df['variables'] = variables
   
    return  coef_df

    
def adjusted_r2(X, y_true, y_pred):   
    print((1-(1-r2_score(y_true, y_pred))*((len(X)-1))/(len(X)-len(X.columns)-1)))

def generate_results_dataset(preds, ci):
    df = pd.DataFrame()
    df['prediction'] = preds
    if ci >= 0:
        df['upper'] = preds + ci
        df['lower'] = preds - ci
    else:
        df['upper'] = preds - ci
        df['lower'] = preds + ci
        
    return df