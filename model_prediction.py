# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, itertools, warnings
import matplotlib.pyplot as plt
#%matplotlib inline
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

#Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#Metrics for evaluation
from sklearn.metrics import confusion_matrix, hinge_loss, matthews_corrcoef, cohen_kappa_score

manual_metric_names = ["prob2d_entropy", "mutual_information", "correl_coefficient", "coexpression"]
pca_components = ["pca-1", "pca-2", "pca-3", "pca-4"]
label_fields = ["model_ID"]#, "group_ID"
labels = [np.arange(1, 17), np.arange(1, 5)]

def load_dataset():
    data_fp = "DATA/model_prediction/2GeneFlex_ML_dataset.csv"
    df = pd.read_csv(data_fp, sep=",", header=0)
    return df

df = load_dataset()
RANDOM_STATE = 42
TEST_SET_RATIO = 0.33

def select_columns(data_fields, label_field):
    x_data = df[data_fields].values.astype(float)
    y_data = df[label_field].values.astype(int)
    return x_data, y_data


def build_classifiers():
    classifier_names = [
        # "LR",
        # "DTN",
        "KNN",
        #"MLP",
        #"GBDT"
    ]

    classifiers = [
        # LogisticRegression(solver='lbfgs', multi_class='multinomial'),
        # DecisionTreeClassifier(max_depth = 3),
        KNeighborsClassifier(n_neighbors=1)#,
        #MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 100, 20), random_state=RANDOM_STATE, max_iter=50000, learning_rate='adaptive')
        #GradientBoostingClassifier(n_estimators=5, learning_rate=.1, max_features=2, max_depth=2,random_state=RANDOM_STATE)
    ]
    return [classifier_names, classifiers]

def prediction(X_train, X_test, y_train, y_test):
    [classifier_names, classifiers] = build_classifiers()
    performances = []
    for cidx, clf_name in enumerate(classifier_names):
        clf = classifiers[cidx].fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "decision_function"):
            pred_decision = clf.decision_function(X_test)
        else:
            pred_decision = clf.predict_proba(X_test)#[:, 1]
        perf = [cohen_kappa_score(y_test, y_pred), hinge_loss(y_test, pred_decision), matthews_corrcoef(y_test, y_pred)]
        print("%s\t cohen_kappa_score: %.2f\t hinge_loss: %.2f\tmatthews_corrcoef:%.2f " % (clf_name, perf[0], perf[1], perf[2]))
        #print(confusion_matrix(y_test, y_pred))
        # performances.append(perf)
    return performances

def model_traning(training_fields, label_field = "model_ID"):
    print("%s\t%s" % (",".join(training_fields), label_field))
    X, y = select_columns(training_fields, label_field)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SET_RATIO, random_state= RANDOM_STATE)
    prediction(X_train, X_test, y_train, y_test)

def looping_each_possible_individual_traning_scheme():
    training_fields = manual_metric_names + pca_components
    for training_field in training_fields:
        for label_field in label_fields:
            model_traning([training_field], label_field)

def train_the_model_using_two_fields():
    combination_metrics = list(itertools.combinations(range(len(manual_metric_names)), 2))
    combination_metric_names = [[manual_metric_names[m1], manual_metric_names[m2]] for (m1, m2) in combination_metrics]
    combination_pca_names = [[pca_components[m1], pca_components[m2]] for (m1, m2) in combination_metrics]
    fields_list = combination_metric_names + combination_pca_names
    for tid, training_fields in enumerate(fields_list):
        for label_field in label_fields:
            model_traning(training_fields, label_field)

def train_the_model_using_four_fields():
    fields_list = [manual_metric_names, pca_components]
    for tid, training_fields in enumerate(fields_list):
        for label_field in label_fields:
            model_traning(training_fields, label_field)

if __name__ == "__main__":
    looping_each_possible_individual_traning_scheme()
    train_the_model_using_two_fields()
    train_the_model_using_four_fields()