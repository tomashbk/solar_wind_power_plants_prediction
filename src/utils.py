import pyhere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""
Global variables
"""
DIR_DATA_RAW = pyhere.here().resolve().joinpath("data", "raw")
DIR_DATA_INTERIM = pyhere.here().resolve().joinpath("data", "interim")
DIR_DATA_EXTERNAL = pyhere.here().resolve().joinpath("data", "external")
SEASONS = {'autumn', 'spring', 'summer', 'winter'}
YEARS = {'2013', '2014', '2015', '2016', '2017', '2018'}
NORTH_HEMISPHERE_MONTHS_SEASONS = dict()
SOUTH_HEMISPHERE_MONTHS_SEASONS = dict()
MONTHS_OF_YEAR = np.array(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])
NORTH_HEMISPHERE_MONTHS_SEASONS["autumn"] = np.array(["OCT", "NOV", "DEC"])
NORTH_HEMISPHERE_MONTHS_SEASONS["winter"] = np.array(["JAN", "FEB", "MAR"])
NORTH_HEMISPHERE_MONTHS_SEASONS["spring"] = np.array(["APR", "MAY", "JUN"])
NORTH_HEMISPHERE_MONTHS_SEASONS["summer"] = np.array(["JUL", "AUG", "SEP"])
SOUTH_HEMISPHERE_MONTHS_SEASONS["spring"] = np.array(["OCT", "NOV", "DEC"])
SOUTH_HEMISPHERE_MONTHS_SEASONS["summer"] = np.array(["JAN", "FEB", "MAR"])
SOUTH_HEMISPHERE_MONTHS_SEASONS["autumn"] = np.array(["APR", "MAY", "JUN"])
SOUTH_HEMISPHERE_MONTHS_SEASONS["winter"] = np.array(["JUL", "AUG", "SEP"])


def make_mi_scores(X, y):
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def get_accuracy(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_val)

    return accuracy

def get_accuracy_knn(n_neighbors, X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors = n_neighbors)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_val)

    return accuracy

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()
    
    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Plot the matrix
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap, ax=ax)