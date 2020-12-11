import random

import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
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
    frac = (len(df_training_1) - len(df_training_0))/len(df_training_0)
    df_training_frac = df_training_0.sample(frac=frac)
    final_df = pd.concat([df_training_0, df_training_frac, df_training_1], axis=0)
    return final_df


def test(normalized_data, algo="K-Means"):
    X_positive = normalized_data[normalized_data["diagnosis"] == 1]
    X_negative = normalized_data[normalized_data["diagnosis"] == 0]

    final_spectral_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_spectral_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    spectr_train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    spectr_test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    for i in range(1, 31):
        # Randomly selecting Labelled Data (with 50% positive and 50% negative samples) in train set

        X_test = pd.concat([X_positive.sample(frac=0.2), X_negative.sample(frac=0.2)])
        X_train = normalized_data.drop(index=X_test.index.tolist())
        X_train = uniform_distribution(X_train)

        dist = {"Malignant": 0, "Benign": 0}
        for item in X_train['diagnosis'].tolist():
            if item == 0:
                dist['Malignant'] += 1
            else:
                dist['Benign'] += 1

        y_test = X_test["diagnosis"]
        y_train = X_train["diagnosis"]

        X_test = X_test.drop(["diagnosis"], axis=1)
        X_train = X_train.drop(["diagnosis"], axis=1)

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        labels = None

        if algo == 'Spectral':
            clustering = SpectralClustering(n_clusters=2, affinity='rbf', n_init=20, random_state=random.randint(20, 200)).fit(X_train)
            labels = pd.DataFrame(clustering.labels_)
        elif algo == 'Hierarchic':
            clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X_train)
            labels = pd.DataFrame(clustering.labels_)
        elif algo == 'K-Means':
            clustering = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                                algorithm='auto').fit(X_train)
            labels = pd.DataFrame(clustering.labels_)

        model = KNeighborsClassifier(n_neighbors=20, weights='distance')
        model.fit(X_train, labels)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        # Accuracy, Precision, Recall, F-score, and AUC of TRAIN SET and TEST SET

        # Accuracy for Train and Test
        spectr_train_results['accuracy'].append(accuracy_score(y_train, y_pred_train))
        spectr_test_results['accuracy'].append(
            accuracy_score(y_test, y_pred))

        # Precision for Train and Test
        spectr_train_results['precision'].append(precision_score(y_train, y_pred_train))
        spectr_test_results['precision'].append(
            precision_score(y_test, y_pred))

        # Recall for Train and Test
        spectr_train_results['recall'].append(recall_score(y_train, y_pred_train))
        spectr_test_results['recall'].append(
            recall_score(y_test, y_pred))

        # F-score for Train and Test
        spectr_train_results['fscore'].append(f1_score(y_train, y_pred_train))
        spectr_test_results['fscore'].append(f1_score(y_test, y_pred))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, y_pred_train)
        area_uc = auc(fpr, tpr)
        spectr_train_results['auc'].append(auc(fpr, tpr))

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        area_uc = auc(fpr, tpr)
        spectr_test_results['auc'].append(auc(fpr, tpr))

        # Average scores (accuracy, precision, recall, F-score, and AUC) for Train set
        final_spectral_train_results['accuracy'] = np.mean(spectr_train_results['accuracy'])
        final_spectral_train_results['precision'] = np.mean(spectr_train_results['precision'])
        final_spectral_train_results['recall'] = np.mean(spectr_train_results['recall'])
        final_spectral_train_results['fscore'] = np.mean(spectr_train_results['fscore'])
        final_spectral_train_results['auc'] = np.mean(spectr_train_results['auc'])

        # Average scores (accuracy, precision, recall, F-score, and AUC) for Test set
        final_spectral_test_results['accuracy'] = np.mean(spectr_test_results['accuracy'])
        final_spectral_test_results['precision'] = np.mean(spectr_test_results['precision'])
        final_spectral_test_results['recall'] = np.mean(spectr_test_results['recall'])
        final_spectral_test_results['fscore'] = np.mean(spectr_test_results['fscore'])
        final_spectral_test_results['auc'] = np.mean(spectr_test_results['auc'])

    print(f"\n {algo} Clustering : Score for Train set:\n")
    print(final_spectral_train_results)

    print(f"\n {algo} Clustering : Score for Test set:\n")
    print(final_spectral_test_results)


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

    #print(dataset.head())
    normalized_df = preprocess_unsupervised(dataset)
    test(normalized_df)
    #k_means = k_means_clustering(normalized_df)
    #self_organizing_map(normalized_df)
    #clustering_visualization(normalized_df)
    #k_means = k_means_clustering(normalized_df)
    #hierarchical_clustering(normalized_df)
    #spectral_clustering(normalized_df, k_means)
    #tsne_clustering(normalized_df)


