import random

import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering


def preprocess_unsupervised(dataset):
    X = dataset.drop(["diagnosis"], axis=1)
    Y = np.where(dataset["diagnosis"] == "M", 0, 1)
    Y = pd.DataFrame(Y)
    Y.columns = ["diagnosis"]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X)
    normalized_data = pd.concat([X, Y], axis=1)
    return normalized_data


def uniform_distribution(df_training):
    df_training_0 = df_training.loc[df_training['diagnosis'] == 0]
    df_training_1 = df_training.loc[df_training['diagnosis'] == 1]
    frac = min(abs((len(df_training_1) - len(df_training_0))/len(df_training_0)), 1)
    df_training_frac = df_training_0.sample(frac=frac)
    final_df = pd.concat([df_training_0, df_training_frac, df_training_1], axis=0)
    return final_df


def tune_paramters(data):
    X_train = data.loc[:, data.columns != 'diagnosis']
    y_train = data['diagnosis']
    model = SpectralClustering()
    my_scorer = make_scorer(accuracy_score, greater_is_better=True)
    parameters = {'affinity': ['rbf'], 'n_clusters': [2], 'gamma': [10, 1, 0.1, 0.01, 0.001],
                  'assign_labels': ['kmeans', 'discretize'], 'n_init': [30]}
    kfolds = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=0)
    clf = GridSearchCV(model, parameters, n_jobs=-1, cv=kfolds, scoring=my_scorer, verbose=True)
    clf.fit(X_train, np.array(y_train))
    best_params = clf.best_params_
    best_score = clf.best_score_
    print("Best parameters : ", best_params)
    print("Best score : ", best_score)


def prediction_model(normalized_data, algo="K-Means"):
    final_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    kfolds = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=0)

    for train, test in kfolds.split(normalized_df):

        # Training set
        training_set = normalized_data.loc[train]
        training_df = uniform_distribution(training_set)
        X_train = training_df.loc[:, training_df.columns != 'diagnosis']
        y_train = training_df['diagnosis']

        # Test set
        test_df = normalized_data.loc[test]
        X_test = test_df.loc[:, test_df.columns != 'diagnosis']
        y_test = test_df['diagnosis']

        # Clustering model
        labels = None
        if algo == 'Spectral':
            clustering = SpectralClustering(n_clusters=2, affinity='rbf', n_init=20, random_state=0).fit(X_train)
            labels = pd.DataFrame(clustering.labels_)
        elif algo == 'Hierarchical':
            labels = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit_predict(X_train)
        elif algo == 'K-Means':
            clustering = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                                algorithm='auto').fit(X_train)
            labels = clustering.labels_
        elif algo == 'GaussianMixture':
            clustering = GaussianMixture(n_components=2).fit(X_train)
            labels = clustering.predict(X_train)
        elif algo == 'Birch':
            labels = Birch(threshold=0.01, n_clusters=2).fit_predict(X_train)

        # Prediction model
        model = KNeighborsClassifier(n_neighbors=20, weights='distance')
        model.fit(X_train, labels)
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        # Accuracy for Train and Test
        train_results['accuracy'].append(accuracy_score(y_train, y_pred_train))
        test_results['accuracy'].append(accuracy_score(y_test, y_pred))

        # Precision for Train and Test
        train_results['precision'].append(precision_score(y_train, y_pred_train))
        test_results['precision'].append(precision_score(y_test, y_pred))

        # Recall for Train and Test
        train_results['recall'].append(recall_score(y_train, y_pred_train))
        test_results['recall'].append(recall_score(y_test, y_pred))

        # F-score for Train and Test
        train_results['fscore'].append(f1_score(y_train, y_pred_train))
        test_results['fscore'].append(f1_score(y_test, y_pred))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, y_pred_train)
        area_uc = auc(fpr, tpr)
        train_results['auc'].append(auc(fpr, tpr))
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        area_uc = auc(fpr, tpr)
        test_results['auc'].append(auc(fpr, tpr))

    # Average scores (accuracy, precision, recall, F-score, and AUC) for Train set
    final_train_results['accuracy'] = np.mean(train_results['accuracy'])
    final_train_results['precision'] = np.mean(train_results['precision'])
    final_train_results['recall'] = np.mean(train_results['recall'])
    final_train_results['fscore'] = np.mean(train_results['fscore'])
    final_train_results['auc'] = np.mean(train_results['auc'])

    # Average scores (accuracy, precision, recall, F-score, and AUC) for Test set
    final_test_results['accuracy'] = np.mean(test_results['accuracy'])
    final_test_results['precision'] = np.mean(test_results['precision'])
    final_test_results['recall'] = np.mean(test_results['recall'])
    final_test_results['fscore'] = np.mean(test_results['fscore'])
    final_test_results['auc'] = np.mean(test_results['auc'])

    print(f"\n {algo} Clustering : Score for Train set:\n")
    print(final_train_results)
    print(f"\n {algo} Clustering : Score for Test set:\n")
    print(final_test_results)


def clustering_visualization(normalized_df):
    data_drop = normalized_df.drop('diagnosis', axis=1)
    X = data_drop.values

    tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
    Y = tsne.fit_transform(X)

    kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                  verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    kY = kmns.fit_predict(X)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(Y[:, 0], Y[:, 1], c=kY, cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('k-means clustering plot')
    ax2.scatter(Y[:, 0], Y[:, 1], c=normalized_df['diagnosis'], cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title('Initial clusters')

    kmns = SpectralClustering(n_clusters=2, gamma=0.5, affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', degree=3,
                              coef0=1, kernel_params=None, n_jobs=1)
    kY = kmns.fit_predict(X)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(Y[:, 0], Y[:, 1], c=kY, cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('Spectral clustering plot')
    ax2.scatter(Y[:, 0], Y[:, 1], c=normalized_df['diagnosis'], cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title('Initial clusters')

    aggC = AgglomerativeClustering(n_clusters=2, linkage='ward')
    kY = aggC.fit_predict(X)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(Y[:, 0], Y[:, 1], c=kY, cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('Hierarchical clustering plot')
    ax2.scatter(Y[:, 0], Y[:, 1], c=normalized_df['diagnosis'], cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title('Initial clusters')


if __name__ == '__main__':
    dataset = pd.read_csv('./data.csv')
    dataset = dataset.drop(["id", "Unnamed: 32"], axis=1)

    """list_diag = dataset['diagnosis'].tolist()
    dict_diag = {"Benign": 0, "Malignant": 0}
    for item in list_diag:
        if item == "M":
            dict_diag['Malignant'] += 1
        else:
            dict_diag['Benign'] += 1
    print(dict_diag)
    plt.bar(dict_diag.keys(), dict_diag.values(), width=0.5)
    plt.ylabel("Distribution of diagnosis")
    plt.show()"""

    normalized_df = preprocess_unsupervised(dataset)
    tune_paramters(normalized_df)
    #prediction_model(normalized_df, algo="Birch")


