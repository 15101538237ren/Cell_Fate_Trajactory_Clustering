# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math, itertools, warnings
import matplotlib.pyplot as plt
#%matplotlib inline
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

#Machine learning models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Metrics for evaluation
from sklearn.metrics import confusion_matrix, hinge_loss, matthews_corrcoef, cohen_kappa_score

manual_metric_names = ["prob2d_entropy", "mutual_information", "correl_coefficient", "coexpression"]
pca_components = ["pca-1", "pca-2", "pca-3", "pca-4"]
label_fields = ["group_ID"]#

def load_dataset():
    data_fp = "DATA/model_prediction/2GeneFlex_ML_dataset.csv"
    df = pd.read_csv(data_fp, sep=",", header=0)
    return df

id_type = label_fields[0]
group_id = True if id_type == "group_ID" else False
def subsampling():
    df = load_dataset()
    ids_valid = [2, 3, 4, 5] if group_id else range(2, 16)
    df = df[df[label_fields[0]].isin(ids_valid)]
    ret_result = np.unique(df[label_fields[0]].values.astype(int), return_counts=True)
    print(ret_result)

    class_labels, class_counts = ret_result
    minimum_data_count_of_class = np.min(class_counts)

    dfs = []
    for class_label in class_labels:
        df_class = df[df[label_fields[0]] == class_label]
        df_under_sampled = df_class.sample(minimum_data_count_of_class)
        dfs.append(df_under_sampled)

    df_uds = pd.concat(dfs, axis=0)
    print(np.unique(df_uds[label_fields[0]], return_counts=True))
    return df_uds
df = subsampling()

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
        # "MLP",
        # "GBDT"
    ]

    classifiers = [
        # LogisticRegression(solver='lbfgs', multi_class='multinomial'),
        # DecisionTreeClassifier(max_depth = 3),
        KNeighborsClassifier(n_neighbors=1)#,
        # MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 16), random_state=RANDOM_STATE, max_iter=5000, learning_rate='adaptive')
        # GradientBoostingClassifier(n_estimators=5, learning_rate=.1, max_features=2, max_depth=2,random_state=RANDOM_STATE)
    ]
    return [classifier_names, classifiers]

def prediction(X_train, X_test, y_train, y_test):
    [classifier_names, classifiers] = build_classifiers()
    for cidx, clf_name in enumerate(classifier_names):
        clf = classifiers[cidx].fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "decision_function"):
            pred_decision = clf.decision_function(X_test)
        else:
            pred_decision = clf.predict_proba(X_test)#[:, 1]
        performances = [cohen_kappa_score(y_test, y_pred), hinge_loss(y_test, pred_decision), matthews_corrcoef(y_test, y_pred)]
        print("%s\t cohen_kappa_score: %.2f\t hinge_loss: %.2f\tmatthews_corrcoef:%.2f " % (clf_name, performances[0], performances[1], performances[2]))
        cm = confusion_matrix(y_test, y_pred)
        return ["%.2f" % item for item in performances], cm

def plot_confusion_matrixs(cms, traning_fields_names, ml_method_name = "KNN",group_id=False,normalize=False):
    id_type = "group_id" if group_id else "model_id"
    classes = [str(item) for item in np.arange(2, 7)] if group_id else [str(item) for item in np.arange(1, 17)]
    fig_fp = "confusion_matrix_for_%s_by_%s.png" % (id_type, ml_method_name)
    cmap = plt.get_cmap('Blues')
    N_COL = 4
    EACH_SUB_FIG_SIZE = 5 if group_id else 12
    N_ROW = int(math.ceil(len(traning_fields_names) / float(N_COL)))
    fig, axs = plt.subplots(N_ROW, N_COL, figsize=(N_COL * EACH_SUB_FIG_SIZE, N_ROW * EACH_SUB_FIG_SIZE))
    vmax_lim = 1500 if group_id else 500
    vmax = 1.0 if normalize else vmax_lim

    for tid, training_fields in enumerate(traning_fields_names):
        row = int(tid / N_COL)
        col = int(tid % N_COL)
        cm = cms[tid]
        ax = axs[row][col]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax= vmax)
        if tid == 0:
            ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=traning_fields_names[tid],
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
    plt.savefig(fig_fp, dpi=200)
def model_traning(training_fields_list, label_fields, ml_method_name="KNN", group_id=False):
    performances = []
    cms = []
    traning_fields_names = []
    for tid, training_fields in enumerate(training_fields_list):
        label_field = label_fields[tid]
        traning_fields_name = " + ".join(training_fields)
        print("%s\t%s" % (traning_fields_name, label_field))
        X, y = select_columns(training_fields, label_field)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SET_RATIO, random_state= RANDOM_STATE)
        performance, cm = prediction(X_train, X_test, y_train, y_test)
        cms.append(cm)
        performance_record = [traning_fields_name, label_field] + performance
        performances.append(performance_record)
        traning_fields_names.append(traning_fields_name)
    plot_confusion_matrixs(cms, traning_fields_names, ml_method_name=ml_method_name,group_id=group_id,normalize=False)
    return np.array(performances)

TRAINING_FIELDS = []
LABEL_FIELDS = []
def looping_each_possible_individual_traning_scheme():
    training_fields = manual_metric_names + pca_components
    for training_field in training_fields:
        for label_field in label_fields:
            TRAINING_FIELDS.append([training_field])
            LABEL_FIELDS.append(label_field)

def train_the_model_using_two_fields():
    combination_metrics = list(itertools.combinations(range(len(manual_metric_names)), 2))
    combination_metric_names = [[manual_metric_names[m1], manual_metric_names[m2]] for (m1, m2) in combination_metrics]
    combination_pca_names = [[pca_components[m1], pca_components[m2]] for (m1, m2) in combination_metrics]
    fields_list = combination_metric_names + combination_pca_names
    for tid, training_fields in enumerate(fields_list):
        for label_field in label_fields:
            TRAINING_FIELDS.append(training_fields)
            LABEL_FIELDS.append(label_field)

def train_the_model_using_four_fields():
    fields_list = [manual_metric_names, pca_components]
    for tid, training_fields in enumerate(fields_list):
        for label_field in label_fields:
            TRAINING_FIELDS.append(training_fields)
            LABEL_FIELDS.append(label_field)

if __name__ == "__main__":
    looping_each_possible_individual_traning_scheme()
    train_the_model_using_two_fields()
    train_the_model_using_four_fields()
    ml_method_name = "KNN"
    performances = model_traning(TRAINING_FIELDS, LABEL_FIELDS, ml_method_name = ml_method_name, group_id=group_id)
    header = "training_field_name, target_label, cohen_kappa_score, hinge_loss, matthews_corrcoef"
    performance_fp = "DATA/model_prediction/%s_%s_Performance.csv" % (ml_method_name, id_type)
    np.savetxt(performance_fp, performances[:], fmt='%s,%s,%s,%s,%s', delimiter="\n", header=header, comments="")