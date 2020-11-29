import hashlib
import math
import random
from copy import deepcopy

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def remove_direct_identifier(df):
    """
    Replaces the unique direct identifier (id) by a hashed and salted id
    :param df: input data frame to transform
    :return: data frame where the id feature was replaced by hashed and salted id
    """
    new_df = deepcopy(df)
    list_id = new_df['id'].tolist()
    new_list_id = []
    for name in list_id:
        name_to_hash = "MargauxGerard" + name.replace(" ", '') + "JeremieBogaert"  # Salting
        new_list_id.append(name_to_hash)
    new_df['id'] = [hashlib.sha1(str.encode(str(i))).hexdigest() for i in new_list_id]  # Hashing
    return new_df


def generalize_data(df):
    """
    Transforms the date of birth of a person by only the year of the birth and the
    Generalizes the zipcode of a person : 26904 -> 26***
    :param df: input data frame to transform
    :return: data frame where the dob feature was replaced by the year of birth
    """
    new_df = deepcopy(df)

    new_list_education = []
    list_education = new_df['education'].tolist()
    list_employment = new_df['employment'].tolist()
    for i in range(len(list_education)):
        if list_employment[i] != 2:
            new_list_education.append(5)
        else:
            new_list_education.append(list_education[i])

    new_df['education'] = new_list_education
    return new_df


def features_encoding(df):
    """
    Encodes some features as numerical values. For example, we will encode gender as : female -> 0, male -> 1
    :param df: input data frame to encode
    :return: data frame where some features were encoded
    """
    new_df = deepcopy(df)
    encoder = LabelEncoder()
    to_be_encoded = ["gender", "education", "employment", "marital_status", "ancestry", "accommodation"]
    for feature in to_be_encoded:
        new_df[feature] = encoder.fit_transform(new_df[feature])
    newDiseaseCol = new_df["disease"]
    for i in range(len( new_df["disease"])):
        if "cancer" in new_df["disease"][i].split(" "):
            newDiseaseCol[i] = "cancer"
    new_df["disease"] = newDiseaseCol
    return new_df


def form_classes(df, flag=True):
    if flag:
        sub_df = df[["employment", "children", "marital_status", "commute_time", "education"]]
        list_tuples = sub_df.to_records()  # Rows of the sub-dataframe into tuples
        dict_classes = dict()
        for (index, employment, children, marital, commute_time, education) in list_tuples:
            dict_classes[(employment, children, marital, commute_time, education)] = dict_classes.get((employment, children, marital, commute_time, education), 0) + 1
    else:
        sub_df = df[["employment", "children", "marital_status", "commute_time", "education", "disease"]]
        list_tuples = sub_df.to_records()  # Rows of the sub-dataframe into tuples
        dict_classes = dict()
        for (index, employment, children, marital, commute_time, education, disease) in list_tuples:
            dict_classes[(employment, children, marital, commute_time, education, disease)] = dict_classes.get((employment, children, marital, commute_time, education, disease), 0) + 1

    dict_classes = dict(sorted(dict_classes.items(), key=lambda x: x[1], reverse=False))
    return dict_classes


def k_anonymization(df, k=2):
    """
    Provides a k-anonymized data frame where unique tuples were removed from the data frame
    :param df: input data frame to transform
    :return: k-anonymized data frame
    """
    index_to_drop = []
    dict_clas = form_classes(df)
    new_df = deepcopy(df)
    for key, val in dict_clas.items():
        if val < k:
            index = new_df.index[(new_df["employment"] == key[0]) & (new_df["children"] == key[1]) &
                                 (new_df["marital_status"] == key[2]) & (new_df["commute_time"] == key[3]) &
                                 (new_df["education"] == key[4])].tolist()
            index_to_drop += index

    new_df = new_df.drop(index_to_drop, axis=0)
    print(len(new_df))
    return new_df


def l_diversity(df, l=2):
    index_to_drop = []
    dict_clas = form_classes(df, flag=False)
    dict_diversity = {}
    new_df = deepcopy(df)
    for key, val in dict_clas.items():
        current_dict = dict_diversity.get(key[0:5], {})
        current_dict[key[5]] = current_dict.get(key[5], 0) + 1
        dict_diversity[key[0:5]] = current_dict

    for key, val in dict_diversity.items():
        if len(val) < l:
            index = new_df.index[(new_df["employment"] == key[0]) & (new_df["children"] == key[1]) &
                                 (new_df["marital_status"] == key[2]) & (new_df["commute_time"] == key[3])
                                 & (new_df["education"] == key[4])].tolist()
            index_to_drop += index

    new_df = new_df.drop(index_to_drop, axis=0)
    print(len(new_df))
    return new_df, dict_diversity


def entropy(df, dict_classes):
    number_of_rows = len(df['id'].tolist())
    result = 0
    for key, val in dict_classes.items():
        prop = val/number_of_rows
        result -= prop*math.log(prop)
    return result


def loss_entropy(initial_df, final_df):
    initial_entropy = entropy(initial_df, form_classes(initial_df))
    final_entropy = entropy(final_df, form_classes(final_df))
    return initial_entropy - final_entropy


def t_closeness(df, t=0.05):
    dict_total_diseases = {}
    new_df = deepcopy(df)
    for i in new_df["disease"]:
        dict_total_diseases[i] = dict_total_diseases.get(i, 0) + 1


    listDisease = list(dict_total_diseases.keys())
    indexMappping = {}
    for i in range(len(listDisease)):
        indexMappping[listDisease[i]] = i

    list_total_diseases = list(dict_total_diseases.values())

    result = 0
    for test in range(1000):
        res = [random.randrange(1, 1000, 1) for j in range(13)]
        new_i = []
        for elem in res:
            val = elem * sum(dict_total_diseases.values()) / sum(res)
            new_i.append(val)
        x = [(i / sum(dict_total_diseases.values())) for i in list(dict_total_diseases.values())]
        if sum(new_i) != 0:
            y = [(i / sum(new_i)) for i in new_i]
        else:
            y = 0
        result += kl_divergence(x, y)
    #print('KL-divergence random : ', result/1000)

    _, dict_diversity = l_diversity(new_df)

    matrixNumberOfDiseases = []
    for i in dict_diversity.values():
        numberOfDiseases = [0 for i in range(len(listDisease))]
        for j in i:
            numberOfDiseases[indexMappping[j]] += i[j]
        matrixNumberOfDiseases.append(numberOfDiseases)
    dist = []
    count = 0
    xs = []
    ys = []
    for i in matrixNumberOfDiseases:
        new_i = []
        for elem in i:
            val = elem*sum(list_total_diseases)/sum(i)
            new_i.append(val)
        #distance = pyemd.emd_samples(new_i, list_total_diseases, normalized=True)
        x = [(i / sum(list_total_diseases)) for i in list_total_diseases]
        y = [(i / sum(new_i)) for i in new_i]
        xs.append(x)
        ys.append(y)
        distance = kl_divergence(x, y)
        dist.append((distance, count))
        count += 1
    dist = sorted(dist)
    dist = [i for i in dist if i[0]>t]
    index_to_drop = []
    count = 0
    for distance, counting in dist:
        key = list(dict_diversity.keys())[counting]
        index = new_df.index[(new_df["employment"] == key[0]) & (new_df["children"] == key[1]) &
                             (new_df["marital_status"] == key[2]) & (new_df["commute_time"] == key[3])
                             & (new_df["education"] == key[4])].tolist()
        index_to_drop += index
        if count == len(dist)-1:
            plt.plot(listDisease, xs[counting], label = "Total distribution")
            plt.plot(listDisease, ys[counting], label = "Class distribution")
            print(ys[counting])
            plt.legend()
            plt.show()
            print(key)
        count +=1

    new_df = new_df.drop(index_to_drop, axis=0)
    return new_df, indexMappping, dict_total_diseases


def kl_divergence(p, q):
    val = 0
    for i in range(len(p)):
        if p[i] != 0 and q[i] != 0:
            val += p[i] * math.log2(p[i] / q[i])
    return val


if __name__ == '__main__':
    original_df = pd.read_csv('dataset-Privacy-Engineering.csv')
    print(original_df.shape)

    encoded_df = features_encoding(original_df)
    hashed_df = remove_direct_identifier(encoded_df)
    pseudonimized_df = generalize_data(hashed_df)
    pseudonimized_df = pseudonimized_df.drop(["gender", "zipcode", "ancestry", "number_vehicles", "accommodation"], axis=1)
    anonymised_df = k_anonymization(pseudonimized_df, k=8)
    diversified_df, _ = l_diversity(anonymised_df, l=6)
    t_close_df, indexMap, old_dict_total_diseases = t_closeness(diversified_df)
    print(f"Loss total: {loss_entropy(original_df, t_close_df)}")
    print(t_close_df.shape)

    """dict_total_diseases = {}
    new_df = deepcopy(t_close_df)
    for i in new_df["disease"]:
        dict_total_diseases[i] = dict_total_diseases.get(i, 0) + 1

    dict_total_diseases_ordered_key = [0] * len(dict_total_diseases)
    dict_total_diseases_ordered_val = [0] * len(dict_total_diseases)
    for i in range(len(dict_total_diseases_ordered_key)):
        key = list(dict_total_diseases.keys())[i]
        dict_total_diseases_ordered_key[indexMap[key]] = list(dict_total_diseases.keys())[i]
        dict_total_diseases_ordered_val[indexMap[key]] = list(dict_total_diseases.values())[i]

    x = [(i/sum(list(old_dict_total_diseases.values()))) for i in list(old_dict_total_diseases.values())]
    y = [(i/sum(dict_total_diseases_ordered_val)) for i in dict_total_diseases_ordered_val]
    print("KL-divergence after t-closeness : ", kl_divergence(x, y))"""

    """ plt.plot(dict_total_diseases_ordered_key, dict_total_diseases_ordered_val, '-')
    plt.plot(dict_total_diseases_ordered_key, dict_total_diseases_ordered_val, 'o')
    plt.xlabel("Diseases")
    plt.ylabel("Distribution in the dataset")
    plt.xticks(rotation=30)
    plt.show()"""

    """list_k = list(range(1, 16, 1))
    list_l = list(range(1, 16, 1))
    final_result = []
    for k in list_k:
        current_result = []
        for l in list_l:
            anonymised_df = k_anonymization(pseudonimized_df, k=k)
            divsersified_df, _ = l_diversity(anonymised_df, l=l)
            close, _, _ = t_closeness(divsersified_df)
            loss = loss_entropy(original_df, close)
            current_result.append(loss)
            print(f"k : {k}  l : {l}  loss : {loss}")
        final_result.append(current_result)

    final_result = np.array(final_result)
    plt.imshow(final_result, extent=[1, 15, 1, 15], origin='lower')
    plt.xlabel("Value of l")
    plt.xticks(list(range(1, 16, 1)), list(range(1, 16, 1)))
    plt.ylabel('Value of k')
    plt.colorbar()"""

    #print(f"Loss between original and pseudominized : {loss_entropy(original_df, pseudonimized_df)}")
    #print(f"Loss between pseudominized and anonymized : {loss_entropy(pseudonimized_df, anonymised_df)}")
    #print(f"Loss between anonymized and divsersified : {loss_entropy(anonymised_df, divsersified_df)}")

    #divsersified_df.to_csv('test2.csv')
