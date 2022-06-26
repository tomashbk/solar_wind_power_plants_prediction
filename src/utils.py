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

# GLOBAL VARIABLES
DIR_DATA_RAW = pyhere.here().resolve().joinpath("data", "raw")
DIR_DATA_INTERIM = pyhere.here().resolve().joinpath("data", "interim")
DIR_DATA_EXTERNAL = pyhere.here().resolve().joinpath("data", "external")
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

    y_pred = model.predict(X_test)
    print(f'{np.around(model.score(X_test, y_test) * 100, 2)}%')
    # If data is unordered in nature (i.e. non - Time series) then shuffle = True is right choice.
    results_cvs = cross_val_score(model, X, y, cv=StratifiedKFold(shuffle = True))
    print(f'{np.around(results_cvs * 100, 2)}(%)')
    print(f'Mean: {np.around(results_cvs.mean() * 100, 2)}%, Standard Deviation: {np.around(results_cvs.std() * 100, 2)}%')
    print(classification_report(y_test, y_pred, target_names=list_target_in_order))
    confusion_matrix_return = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix_return, annot=True, fmt = 'g')