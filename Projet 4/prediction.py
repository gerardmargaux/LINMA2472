import random

import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, make_scorer, \
    euclidean_distances, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.svm import SVC
import tensorflow as tf


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


def tune_paramters(X_train, labels, algo='KNN'):
    if algo == 'KNN':
        model = KNeighborsClassifier()
        parameters = {'n_neighbors': [5, 10, 15, 20, 30, 50, 100, 200, 500], 'weights': ['distance', 'uniform'],
                      'p': [1, 2, 3, 4, 5], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    else:
        model = SVC()
        parameters = {'C': [1, 10, 50, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [3, 4, 5],
                      'random_state': [0], 'shrinking': [True, False]}

    my_scorer = make_scorer(accuracy_score, greater_is_better=False)
    kfolds = sklearn.model_selection.KFold(n_splits=10, shuffle=True)
    clf = GridSearchCV(model, parameters, n_jobs=-1, cv=kfolds, scoring=my_scorer, verbose=True)
    clf.fit(X_train, labels)
    best_params = clf.best_params_
    best_score = clf.best_score_
    print(best_params)
    print(best_score)
    return best_params


def my_accuracy_score(y_true, y_pred):
    if accuracy_score(y_true, y_pred) >= 0.5:
        return accuracy_score(y_true, y_pred)
    else:
        return accuracy_score(y_true, 1 - y_pred)


def my_precision_score(y_true, y_pred):
    if accuracy_score(y_true, y_pred) >= 0.5:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, 1 - y_pred)


def my_recall_score(y_true, y_pred):
    if accuracy_score(y_true, y_pred) >= 0.5:
        return recall_score(y_true, y_pred)
    else:
        return recall_score(y_true, 1-y_pred)


def my_f1_score(y_true, y_pred):
    if accuracy_score(y_true, y_pred) >= 0.5:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, 1 - y_pred)


def my_roc_curve(y_true, y_pred):
    if accuracy_score(y_true, y_pred) >= 0.5:
        return roc_curve(y_true, y_pred)
    else:
        return roc_curve(y_true, 1-y_pred)


def prediction_model(normalized_data, algo="K-Means", modelType="Keras", inverse=False):
    final_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    training_df = deepcopy(normalized_data)
    X_train = normalized_data.loc[:, normalized_data.columns != 'diagnosis']
    y_train = normalized_data.loc[:, normalized_data.columns == 'diagnosis']

    # Clustering model
    labels = None
    if algo == 'Spectral':
        clustering = SpectralClustering(n_clusters=2, affinity='rbf', gamma=0.01, n_init=30, random_state=0).fit(
            X_train)
        labels = pd.DataFrame(clustering.labels_).to_numpy().ravel()
    elif algo == 'Hierarchical':
        labels = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit_predict(X_train)
    elif algo == 'K-Means':
        clustering = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                            verbose=0, random_state=None, copy_x=True,
                            algorithm='auto').fit(X_train)
        labels = pd.DataFrame(clustering.labels_).to_numpy().ravel()
    elif algo == 'GaussianMixture':
        clustering = GaussianMixture(n_components=2).fit(X_train)
        labels = clustering.predict(X_train)
    elif algo == 'Birch':
        labels = Birch(threshold=0.01, n_clusters=2).fit_predict(X_train)
    elif algo == 'Supervised':
        labels = y_train

    if inverse:
        new_labels = []
        for item in labels:
            new_labels.append(1 - item)
        training_df['labels'] = new_labels
    else:
        training_df['labels'] = labels

    # Prediction model
    uniform_training = uniform_distribution(training_df)
    X_train = uniform_training.loc[:, (uniform_training.columns != 'diagnosis') & (uniform_training.columns != 'labels')]
    model = None
    if modelType == "KNN":
        best_params = tune_paramters(X_train, uniform_training['labels'], algo='KNN')
        model = KNeighborsClassifier(**best_params)
    elif modelType == "Keras":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation='linear'))
        model.add(tf.keras.layers.Dropout(0.6))
        model.add(tf.keras.layers.Dense(64, activation='linear'))
        model.add(tf.keras.layers.Dropout(0.6))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.6))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    elif modelType == "SVM":
        best_params = tune_paramters(X_train, uniform_training['labels'], algo='SVM')
        model = SVC(**best_params)

    kfolds = sklearn.model_selection.KFold(n_splits=10, shuffle=True)
    for train, test in kfolds.split(normalized_df):
        # Training set
        train_df = training_df.loc[train]
        train_df = uniform_distribution(train_df)
        X_train = train_df.loc[:, (train_df.columns != 'diagnosis') & (train_df.columns != 'labels')]
        y_labels = train_df['labels']
        y_train = train_df['diagnosis']

        # Test set
        test_df = training_df.loc[test]
        X_test = test_df.loc[:, (test_df.columns != 'diagnosis') & (test_df.columns != 'labels')]
        y_test = test_df['diagnosis']

        # Prediction model
        if modelType == "KNN" or modelType == "SVM":
            model.fit(X_train, y_labels)
        else:
            model.fit(X_train, y_labels, batch_size=32, epochs=200)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        new_y_pred_train = []
        for item in y_pred_train:
            if item < 0.5:
                new_y_pred_train.append(0)
            else:
                new_y_pred_train.append(1)

        new_y_pred = []
        for item in y_pred:
            if item < 0.5:
                new_y_pred.append(0)
            else:
                new_y_pred.append(1)

        # Accuracy for Train and Test
        train_results['accuracy'].append(accuracy_score(y_train, new_y_pred_train))
        test_results['accuracy'].append(accuracy_score(y_test, new_y_pred))

        # Precision for Train and Test
        train_results['precision'].append(precision_score(y_train, new_y_pred_train))
        test_results['precision'].append(precision_score(y_test, new_y_pred))

        # Recall for Train and Test
        train_results['recall'].append(recall_score(y_train, new_y_pred_train))
        test_results['recall'].append(recall_score(y_test, new_y_pred))

        # F-score for Train and Test
        train_results['fscore'].append(f1_score(y_train, new_y_pred_train))
        test_results['fscore'].append(f1_score(y_test, new_y_pred))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, new_y_pred_train)
        area_uc = auc(fpr, tpr)
        train_results['auc'].append(auc(fpr, tpr))
        fpr, tpr, _ = roc_curve(y_test, new_y_pred)
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

    print(f"\n {algo} Clustering, {modelType} Model : Score for Train set:\n")
    print(final_train_results)

    print(f"\n {algo} Clustering, {modelType} Model : Score for Test set:\n")
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


def distance_matrix(normalized_df, compare_with_true=True):
    labels_true = normalized_df['diagnosis'].tolist()
    data_drop = normalized_df.drop('diagnosis', axis=1)
    X = data_drop.values

    clustering = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                  verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    labels_kmeans = clustering.fit_predict(X)

    clustering = SpectralClustering(n_clusters=2, gamma=0.5, affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', degree=3,
                              coef0=1, kernel_params=None, n_jobs=1)
    labels_spectral = clustering.fit_predict(X)

    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels_hierarchical = clustering.fit_predict(X)

    clustering = GaussianMixture(n_components=2).fit(X)
    labels_gaussian = clustering.predict(X)

    labels_birch = Birch(threshold=0.01, n_clusters=2).fit_predict(X)

    labels = [labels_kmeans, labels_spectral, labels_hierarchical, labels_gaussian, labels_birch]

    if compare_with_true:
        res = []
        for i in labels:
            val = adjusted_rand_score(labels_true, i)
            res.append(val)
        res = np.expand_dims(res, axis=0)
        plt.imshow(res, origin='lower')
        plt.xticks([0, 1, 2, 3, 4], ['K-Means', 'Spectral', 'Hierarchical', 'GaussianMixture', 'Birch'])
        plt.yticks([0], ['Original clustering'])
        plt.colorbar()
    else:
        final_result = []
        for i in labels:
            res = []
            for j in labels:
                val = adjusted_rand_score(i, j)
                res.append(val)
            final_result.append(res)
        final_result = np.array(final_result)
        plt.imshow(final_result, extent=[-1, 1.5, -1, 1.5], origin='lower')
        plt.xticks([-0.75, -0.25, 0.25, 0.75, 1.25], ['K-Means', 'Spectral', 'Hierarchical', 'GaussianMixture', 'Birch'])
        plt.yticks([-0.75, -0.25, 0.25, 0.75, 1.25], ['K-Means', 'Spectral', 'Hierarchical', 'GaussianMixture', 'Birch'])
        plt.colorbar()


if __name__ == '__main__':
    dataset = pd.read_csv('./data.csv')
    dataset = dataset.drop(["id", "Unnamed: 32"], axis=1)

    normalized_df = preprocess_unsupervised(dataset)
    prediction_model(normalized_df, algo="K-Means", modelType="KNN", inverse=False)
    #distance_matrix(normalized_df, compare_with_true=True)

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
