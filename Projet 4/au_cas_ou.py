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


def k_means_clustering(normalized_data):
    # Taking Positive class=Benign and Negative class=Malignant

    X_positive = normalized_data[normalized_data["diagnosis"] == 1]
    X_negative = normalized_data[normalized_data["diagnosis"] == 0]
    final_unsupervised_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_unsupervised_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    unsupr_train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    unsupr_test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    for i in range(1, 31):
        # Randomly selecting Labelled Data (with 50% positive and 50% negative samples) in train set

        X_test = pd.concat([X_positive.sample(frac=0.2), X_negative.sample(frac=0.2)])
        X_train = normalized_data.drop(index=X_test.index.tolist())

        y_test = X_test["diagnosis"]
        y_train = X_train["diagnosis"]

        X_test = X_test.drop(["diagnosis"], axis=1)
        X_train = X_train.drop(["diagnosis"], axis=1)

        X_label, X_unlabel, y_label, y_unlabel = train_test_split(X_train, y_train, test_size=0.50, stratify=y_train)

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Computing the centers of two clusters
        # k_means = KMeans(n_clusters=2, init='k-means++', random_state=random.randint(20, 200), n_init=20)#.
        k_means = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                         precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                         algorithm='auto').fit(X_train)
        clus_dist = k_means.transform(X_train)

        # Finding the closest 30 data points to each center
        nn = NearestNeighbors(n_neighbors=30, algorithm='brute').fit(X_train)
        distances, indices = nn.kneighbors(k_means.cluster_centers_)

        # Reading the true lables of the 30 datapoints
        clus_0 = y_train.loc[indices[0]]
        clus_1 = y_train.loc[indices[1]]

        # Taking a majority poll with these 30 points which becomes the label
        # predicted by k-means for the members of each cluster.
        maj_poll_clus_0 = clus_0.value_counts()
        maj_poll_clus_1 = clus_1.value_counts()

        # Finding the labels provided by K_means
        pred_labels = k_means.labels_
        pred_labels = pd.DataFrame(pred_labels)

        pred_labels_0 = deepcopy(pred_labels.loc[pred_labels[0] == 0, :])
        pred_labels_1 = deepcopy(pred_labels.loc[pred_labels[0] == 1, :])

        # Compare the labels provided by K-means with the true labels of the training data
        max_index_0 = maj_poll_clus_0.idxmax()
        max_index_1 = maj_poll_clus_1.idxmax()
        pred_labels_0['class'] = max_index_0
        pred_labels_1['class'] = max_index_1
        final_pred_y = pd.concat([pred_labels_0['class'], pred_labels_1['class']], axis=0)
        final_pred_y = final_pred_y.sort_index()

        # Testing the final Model on Test data

        unsupervised_test_results = pd.DataFrame()
        unsupervised_test_results["True_y"] = y_test
        unsupervised_test_results["Pred_y"] = k_means.predict(X_test)

        # Accuracy, Precision, Recall, F-score, and AUC of TRAIN SET and TEST SET
        # print(f"Longueur true values : {len(y_train)} / Longueur predicted : {len(final_pred_y)}")

        # Accuracy for Train and Test
        unsupr_train_results['accuracy'].append(accuracy_score(y_train, final_pred_y))
        unsupr_test_results['accuracy'].append(
            accuracy_score(unsupervised_test_results["True_y"], unsupervised_test_results["Pred_y"]))

        # Precision for Train and Test
        unsupr_train_results['precision'].append(precision_score(y_train, final_pred_y))
        unsupr_test_results['precision'].append(
            precision_score(unsupervised_test_results["True_y"], unsupervised_test_results["Pred_y"]))

        # Recall for Train and Test
        unsupr_train_results['recall'].append(recall_score(y_train, final_pred_y))
        unsupr_test_results['recall'].append(
            recall_score(unsupervised_test_results["True_y"], unsupervised_test_results["Pred_y"]))

        # F-score for Train and Test
        unsupr_train_results['fscore'].append(f1_score(y_train, final_pred_y))
        unsupr_test_results['fscore'].append(
            f1_score(unsupervised_test_results["True_y"], unsupervised_test_results["Pred_y"]))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, final_pred_y)
        area_uc = auc(fpr, tpr)
        unsupr_train_results['auc'].append(auc(fpr, tpr))

        fpr, tpr, _ = roc_curve(unsupervised_test_results["True_y"], unsupervised_test_results["Pred_y"])
        area_uc = auc(fpr, tpr)
        unsupr_test_results['auc'].append(auc(fpr, tpr))

        # Average scores (accuracy, precision, recall, F-score, and AUC) for Train set
        final_unsupervised_train_results['accuracy'] = np.mean(unsupr_train_results['accuracy'])
        final_unsupervised_train_results['precision'] = np.mean(unsupr_train_results['precision'])
        final_unsupervised_train_results['recall'] = np.mean(unsupr_train_results['recall'])
        final_unsupervised_train_results['fscore'] = np.mean(unsupr_train_results['fscore'])
        final_unsupervised_train_results['auc'] = np.mean(unsupr_train_results['auc'])

        # Average scores (accuracy, precision, recall, F-score, and AUC) for Test set
        final_unsupervised_test_results['accuracy'] = np.mean(unsupr_test_results['accuracy'])
        final_unsupervised_test_results['precision'] = np.mean(unsupr_test_results['precision'])
        final_unsupervised_test_results['recall'] = np.mean(unsupr_test_results['recall'])
        final_unsupervised_test_results['fscore'] = np.mean(unsupr_test_results['fscore'])
        final_unsupervised_test_results['auc'] = np.mean(unsupr_test_results['auc'])

    """print("\n K-means Clustering : Average Score over 30 Runs for Train set:\n")
    print(final_unsupervised_train_results)

    print("\n K-means Clustering : Average Score over 30 Runs for Test set:\n")
    print(final_unsupervised_test_results)"""

    return k_means


def spectral_clustering(normalized_data, k_means):
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

        y_test = X_test["diagnosis"]
        y_train = X_train["diagnosis"]

        X_test = X_test.drop(["diagnosis"], axis=1)
        X_train = X_train.drop(["diagnosis"], axis=1)

        X_label, X_unlabel, y_label, y_unlabel = train_test_split(X_train, y_train, test_size=0.50, stratify=y_train)

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Computing the centers of two clusters
        sp_clus = SpectralClustering(n_clusters=2, affinity='rbf', n_init=20, random_state=random.randint(20, 200)).fit(
            X_train)
        sp_clus_labels = pd.DataFrame(sp_clus.labels_)
        print(sp_clus_labels)

        sp_clus_label_0 = sp_clus_labels[sp_clus_labels[0] == 0].index
        sp_clus_label_1 = sp_clus_labels[sp_clus_labels[0] == 1].index

        sp_clus_0 = X_train.iloc[sp_clus_label_0, :]
        sp_clus_1 = X_train.iloc[sp_clus_label_1, :]

        center_0 = sp_clus_0.mean(axis=0)
        center_1 = sp_clus_1.mean(axis=0)

        centers = pd.DataFrame()
        centers[0] = center_0
        centers[1] = center_1
        # print(centers.T)

        # Finding the closest 30 data points to each center
        nn = NearestNeighbors(n_neighbors=30, algorithm='brute').fit(X_train)
        distances, indices = nn.kneighbors(centers.T)

        # Reading the true lables of the 30 datapoints
        sp_clus_0 = y_train.loc[indices[0]]
        sp_clus_1 = y_train.loc[indices[1]]

        # Taking a majority poll with these 30 points
        maj_poll_clus_0 = sp_clus_0.value_counts()
        maj_poll_clus_1 = sp_clus_1.value_counts()

        # Finding the labels provided by K_means
        pred_labels_spec = k_means.labels_
        pred_labels_spec = pd.DataFrame(pred_labels_spec)

        pred_labels_spec_0 = deepcopy(pred_labels_spec.loc[pred_labels_spec[0] == 0, :])
        pred_labels_spec_1 = deepcopy(pred_labels_spec.loc[pred_labels_spec[0] == 1, :])

        # Compare the labels provided by K-means with the true labels of the training data

        max_index_spec_0 = maj_poll_clus_0.idxmax()
        max_index_spec_1 = maj_poll_clus_1.idxmax()
        pred_labels_spec_0['class'] = max_index_spec_0
        pred_labels_spec_1['class'] = max_index_spec_1
        final_pred_spec_y = pd.concat([pred_labels_spec_0['class'], pred_labels_spec_1['class']], axis=0)
        final_pred_spec_y = final_pred_spec_y.sort_index()

        # Testing the final Model on Test data

        spectral_test_results = pd.DataFrame()
        spectral_test_results["True_y"] = y_test
        spectral_test_results["Pred_y"] = sp_clus.fit_predict(X_test)

        # Accuracy, Precision, Recall, F-score, and AUC of TRAIN SET and TEST SET

        # Accuracy for Train and Test
        spectr_train_results['accuracy'].append(accuracy_score(y_train, final_pred_spec_y))
        spectr_test_results['accuracy'].append(
            accuracy_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # Precision for Train and Test
        spectr_train_results['precision'].append(precision_score(y_train, final_pred_spec_y))
        spectr_test_results['precision'].append(
            precision_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # Recall for Train and Test
        spectr_train_results['recall'].append(recall_score(y_train, final_pred_spec_y))
        spectr_test_results['recall'].append(
            recall_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # F-score for Train and Test
        spectr_train_results['fscore'].append(f1_score(y_train, final_pred_spec_y))
        spectr_test_results['fscore'].append(f1_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, final_pred_spec_y)
        area_uc = auc(fpr, tpr)
        spectr_train_results['auc'].append(auc(fpr, tpr))

        fpr, tpr, _ = roc_curve(spectral_test_results["True_y"], spectral_test_results["Pred_y"])
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

    print("\n Spectral Clustering : Average Score over 30 Runs for Train set:\n")
    print(final_spectral_train_results)

    print("\n Spectral Clustering : Average Score over 30 Runs for Test set:\n")
    print(final_spectral_test_results)


def uniform_distribution(df_training):
    df_training_0 = df_training.loc[df_training['diagnosis'] == 0]
    df_training_1 = df_training.loc[df_training['diagnosis'] == 1]
    frac = (len(df_training_1) - len(df_training_0)) / len(df_training_0)
    df_training_frac = df_training_0.sample(frac=frac)
    final_df = pd.concat([df_training_0, df_training_frac, df_training_1], axis=0)
    return final_df


def hierarchical_clustering(normalized_data):
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

        X_label, X_unlabel, y_label, y_unlabel = train_test_split(X_train, y_train, test_size=0.50, stratify=y_train)

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # Computing the centers of two clusters
        sp_clus = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X_train)
        sp_clus_labels = pd.DataFrame(sp_clus.labels_)

        sp_clus_label_0 = sp_clus_labels[sp_clus_labels[0] == 0].index
        sp_clus_label_1 = sp_clus_labels[sp_clus_labels[0] == 1].index

        sp_clus_0 = X_train.iloc[sp_clus_label_0, :]
        sp_clus_1 = X_train.iloc[sp_clus_label_1, :]

        center_0 = sp_clus_0.mean(axis=0)
        center_1 = sp_clus_1.mean(axis=0)

        centers = pd.DataFrame()
        centers[0] = center_0
        centers[1] = center_1
        # print(centers.T)

        # Finding the closest 30 data points to each center
        nn = NearestNeighbors(n_neighbors=50).fit(X_train)
        distances, indices = nn.kneighbors(centers.T)

        # Reading the true lables of the 30 datapoints
        sp_clus_0 = y_train.loc[indices[0]]
        sp_clus_1 = y_train.loc[indices[1]]

        # Taking a majority poll with these 30 points
        maj_poll_clus_0 = sp_clus_0.value_counts()
        maj_poll_clus_1 = sp_clus_1.value_counts()

        # Finding the labels provided by K_means
        pred_labels_spec = sp_clus.labels_
        pred_labels_spec = pd.DataFrame(pred_labels_spec)

        pred_labels_spec_0 = deepcopy(pred_labels_spec.loc[pred_labels_spec[0] == 0, :])
        pred_labels_spec_1 = deepcopy(pred_labels_spec.loc[pred_labels_spec[0] == 1, :])

        # Compare the labels provided by K-means with the true labels of the training data

        max_index_spec_0 = maj_poll_clus_0.idxmax()
        max_index_spec_1 = maj_poll_clus_1.idxmax()
        pred_labels_spec_0['class'] = max_index_spec_0
        pred_labels_spec_1['class'] = max_index_spec_1
        final_pred_spec_y = pd.concat([pred_labels_spec_0['class'], pred_labels_spec_1['class']], axis=0)
        final_pred_spec_y = final_pred_spec_y.sort_index()

        # Testing the final Model on Test data

        spectral_test_results = pd.DataFrame()
        spectral_test_results["True_y"] = y_test
        spectral_test_results["Pred_y"] = sp_clus.fit_predict(X_test)

        # Accuracy, Precision, Recall, F-score, and AUC of TRAIN SET and TEST SET

        # Accuracy for Train and Test
        spectr_train_results['accuracy'].append(accuracy_score(y_train, final_pred_spec_y))
        spectr_test_results['accuracy'].append(
            accuracy_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # Precision for Train and Test
        spectr_train_results['precision'].append(precision_score(y_train, final_pred_spec_y))
        spectr_test_results['precision'].append(
            precision_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # Recall for Train and Test
        spectr_train_results['recall'].append(recall_score(y_train, final_pred_spec_y))
        spectr_test_results['recall'].append(
            recall_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # F-score for Train and Test
        spectr_train_results['fscore'].append(f1_score(y_train, final_pred_spec_y))
        spectr_test_results['fscore'].append(f1_score(spectral_test_results["True_y"], spectral_test_results["Pred_y"]))

        # AUC for Train and Test
        fpr, tpr, _ = roc_curve(y_train, final_pred_spec_y)
        area_uc = auc(fpr, tpr)
        spectr_train_results['auc'].append(auc(fpr, tpr))

        fpr, tpr, _ = roc_curve(spectral_test_results["True_y"], spectral_test_results["Pred_y"])
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

    print("\n Hierarchical Clustering : Average Score over 30 Runs for Train set:\n")
    print(final_spectral_train_results)

    print("\n Hierarchical Clustering : Average Score over 30 Runs for Test set:\n")
    print(final_spectral_test_results)


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
        # X_train = uniform_distribution(X_train)

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

        if algo == 'Spectral':
            clustering = SpectralClustering(n_clusters=2, affinity='rbf', n_init=20,
                                            random_state=random.randint(20, 200)).fit(X_train)
            labels = pd.DataFrame(clustering.labels_)
        elif algo == 'Hierarchic':
            clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X_train)
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

    print("\n Hierarchical Clustering : Average Score over 30 Runs for Train set:\n")
    print(final_spectral_train_results)

    print("\n Hierarchical Clustering : Average Score over 30 Runs for Test set:\n")
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

    # print(dataset.head())
    normalized_df = preprocess_unsupervised(dataset)
    test(normalized_df)
    # k_means = k_means_clustering(normalized_df)
    # self_organizing_map(normalized_df)
    # clustering_visualization(normalized_df)
    # k_means = k_means_clustering(normalized_df)
    # hierarchical_clustering(normalized_df)
    # spectral_clustering(normalized_df, k_means)
    # tsne_clustering(normalized_df)


