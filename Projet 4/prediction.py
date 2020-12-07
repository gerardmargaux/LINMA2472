import random

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering


def preprocess_unsupervised(full_df):
    scaler = MinMaxScaler()
    X = full_df.drop(["diagnosis"], axis=1)
    y = pd.DataFrame(np.where(full_df["diagnosis"] == "M", 0, 1))  # Benign = 1 / Malignant = 0
    y.columns = ["diagnosis"]
    X = pd.DataFrame(scaler.fit_transform(X))
    normalized_data = pd.concat([X, y], axis=1)
    return normalized_data


def k_means_clustering(normalized_data):
    final_unsupervised_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_unsupervised_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    unsupr_train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    unsupr_test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    X_positive = normalized_data[normalized_data["diagnosis"] == 1]
    X_negative = normalized_data[normalized_data["diagnosis"] == 0]

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

        X_train = X_train.dropna(axis=1)
        X_test = X_test.dropna(axis=1)

        k_means = KMeans(n_clusters=2, init='k-means++', random_state=random.randint(20, 200), n_init=20).fit(X_train)
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
        pred_labels.columns = ['predicted label']

        pred_labels_0 = pred_labels[pred_labels['predicted label'] == 0]
        pred_labels_1 = pred_labels[pred_labels['predicted label'] == 1]

        # Compare the labels provided by K-means with the true labels of the training data
        max_index_0 = np.argmax(maj_poll_clus_0)
        max_index_1 = np.argmax(maj_poll_clus_1)
        pred_labels_0['class'] = max_index_0
        pred_labels_1['class'] = max_index_1

        final_pred_y = pd.concat([pred_labels_0['class'], pred_labels_1['class']], axis=0)
        final_pred_y = final_pred_y.sort_index()

        # Testing the final Model on Test data
        unsupervised_test_results = pd.DataFrame()
        unsupervised_test_results["True_y"] = y_test
        unsupervised_test_results["Pred_y"] = k_means.predict(X_test)

        # Accuracy, Precision, Recall, F-score, and AUC of TRAIN SET and TEST SET

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

    print("\n Average Score over 30 Runs for Train set:\n")
    print(final_unsupervised_train_results)

    print("\n Average Score over 30 Runs for Test set:\n")
    print(final_unsupervised_test_results)

    return k_means


def spectral_clustering(normalized_data, k_means):
    final_spectral_train_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    final_spectral_test_results = {p: None for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    spectr_train_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}
    spectr_test_results = {p: [] for p in ['accuracy', 'precision', 'recall', 'fscore', 'auc']}

    X_positive = normalized_data[normalized_data["diagnosis"] == 1]
    X_negative = normalized_data[normalized_data["diagnosis"] == 0]

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

        X_train = X_train.dropna(axis=1)
        X_test = X_test.dropna(axis=1)

        # Computing the centers of two clusters
        sp_clus = SpectralClustering(n_clusters=2, affinity='rbf', n_init=20, random_state=random.randint(20, 200)).fit(
            X_train)
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

        pred_labels_spec_0 = pred_labels_spec[pred_labels_spec[0] == 0]
        pred_labels_spec_1 = pred_labels_spec[pred_labels_spec[0] == 1]

        # Compare the labels provided by K-means with the true labels of the training data

        max_index_spec_0 = np.argmax(maj_poll_clus_0)
        max_index_spec_1 = np.argmax(maj_poll_clus_1)
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

    print("\n Average Score over 30 Runs for Train set:\n")
    print(final_spectral_train_results)

    print("\n Average Score over 30 Runs for Test set:\n")
    print(final_spectral_test_results)


if __name__ == '__main__':
    df = pd.read_csv('./data.csv')
    df = df.drop('id', axis=1)
    normalized_df = preprocess_unsupervised(df)
    k_means = k_means_clustering(normalized_df)
    spectral_clustering(normalized_df, k_means)


