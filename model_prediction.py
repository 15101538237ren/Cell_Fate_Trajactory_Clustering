# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, re, time
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score

manual_metric_names = ["prob2d_entropy", "mutual_information", "correl_coefficient", "coexpression"]
pca_components = ["pca-1", "pca-2", "pca-3", "pca-4"]
label_fields = ["model_ID", "group_ID"]

def load_dataset():
    data_fp = "DATA/model_prediction/2GeneFlex_ML_dataset.csv"
    df = pd.read_csv(data_fp, sep=",", header=0)
    return df

df = load_dataset()
RANDOM_STATE = 42
TEST_SET_RATIO = 0.33
def select_columns(data_fields, label_field):
    x_data = df[data_fields]
    y_data = df[label_field]
    return x_data, y_data


def build_classifiers():
    classifier_names = [
        "LR",
        #"KNN",
        #"MLP",
        #"GBDT"
    ]

    classifiers = [
        LogisticRegression(C=1, penalty='l1', max_iter=10000),
        # KNeighborsClassifier(n_neighbors=1)#,
        #MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 100, 20), random_state=RANDOM_STATE, max_iter=50000, learning_rate='adaptive')
        #GradientBoostingClassifier(n_estimators=5, learning_rate=.1, max_features=2, max_depth=2,random_state=RANDOM_STATE)
    ]
    return [classifier_names, classifiers]

def prediction(X_train, X_test, y_train,y_test):
    [classifier_names, classifiers] = build_classifiers()
    performances = []
    for cidx, clf_name in enumerate(classifier_names):
        clf = classifiers[cidx].fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
        else:
            Z = clf.predict_proba(X_test)[:, 1]
        fpr_gb, tpr_gb, _ = roc_curve(y_test, Z)
        roc = auc(fpr_gb, tpr_gb)
        perf = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),
                precision_score(y_test, y_pred), f1_score(y_test, y_pred), roc]
        print("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (clf_name, accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),
                                                    precision_score(y_test, y_pred), f1_score(y_test, y_pred), roc))
        performances.append(perf)
    return performances

def model_traning(training_fields, label_field = "model_ID"):
    print("using %s to predict %s" % (",".join(training_fields), label_field))
    X, y = select_columns(training_fields, label_field)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SET_RATIO, random_state= RANDOM_STATE)
    prediction(X_train, X_test, y_train, y_test)

def looping_each_possible_individual_traning_scheme():
    training_fields = manual_metric_names + pca_components
    for training_field in training_fields:
        for label_field in label_fields:
            model_traning([training_field], label_field)

def train_the_model_using_four_fields():
    fields_list = [manual_metric_names, pca_components]
    for tid, training_fields in enumerate(fields_list):
        for label_field in label_fields:
            model_traning(training_fields, label_field)

if __name__ == "__main__":
    looping_each_possible_individual_traning_scheme()
    train_the_model_using_four_fields()