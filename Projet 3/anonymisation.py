import hashlib
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat
from sklearn.preprocessing import LabelEncoder


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

    list_dob = new_df['dob'].tolist()
    new_list_dob = [list_dob[i].split("/")[1] for i in range(len(list_dob))]
    new_list_dob = [int(int(new_list_dob[i])//25)*25 for i in range(len(new_list_dob))]
    new_df['dob'] = new_list_dob

    list_zipcode = new_df['zipcode'].tolist()
    new_df['zipcode'] = [(int(list_zipcode[i])//2000)*2000 for i in range(len(list_zipcode))]
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
    return new_df


def form_classes(df, flag=True):
    if flag:
        sub_df = df[["dob", "zipcode"]]
        list_tuples = sub_df.to_records()  # Rows of the sub-dataframe into tuples
        dict_classes = dict()
        for (index, dob, zipcode) in list_tuples:
            dict_classes[(dob, zipcode)] = dict_classes.get((dob, zipcode), 0) + 1
        dict_classes = dict(sorted(dict_classes.items(), key=lambda x: x[1], reverse=False))
    else:
        sub_df = df[["dob", "zipcode", "disease"]]
        list_tuples = sub_df.to_records()  # Rows of the sub-dataframe into tuples
        dict_classes = dict()
        for (index, dob, zipcode, disease) in list_tuples:
            dict_classes[(dob, zipcode, disease)] = dict_classes.get((dob, zipcode, disease), 0) + 1

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
            index = new_df.index[(new_df["dob"] == key[0]) & (new_df["zipcode"] == key[1])].tolist()
            index_to_drop += index

    new_df = new_df.drop(index_to_drop, axis=0)
    return new_df


def l_diversity(df, l=2, modify_df = True):
    index_to_drop = []
    dict_clas = form_classes(df, flag=False)
    dict_diversity = {}
    new_df = deepcopy(df)
    sum = 0
    for key, val in dict_clas.items():
        dict_diversity[key[0:2]] = dict_diversity.get(key[0:2], {})
        dict_diversity[key[0:2]][key[2]] = dict_diversity[key[0:2]].get(key[2], 0) + val

    if modify_df:
        for key, val in dict_diversity.items():
            if len(val) < l:
                index = new_df.index[(new_df["dob"] == key[0]) & (new_df["zipcode"] == key[1])].tolist()
                index_to_drop += index

        new_df = new_df.drop(index_to_drop, axis=0)
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

def t_closeness(df, t=2):
    dict_total_diseases = {}
    for i in df["disease"]:
        dict_total_diseases[i] = dict_total_diseases.get(i,0) + 1

    plt.plot(dict_total_diseases.keys(),dict_total_diseases.values())
    plt.show()

    listDisease = list(dict_total_diseases.keys())
    indexMappping = {}
    for i in range(len(listDisease)):
        indexMappping[listDisease[i]] = i

    list_total_diseases = list(dict_total_diseases.values())

    _, dict_diversity = l_diversity(df, modify_df = False)

    matrixNumberOfDiseases = []
    for i in dict_diversity.values():
        numberOfDiseases = [0 for i in range(len(listDisease))]
        for j in i:
            numberOfDiseases[indexMappping[j]] += i[j]
        matrixNumberOfDiseases.append(numberOfDiseases)

    for i in matrixNumberOfDiseases:
        distance = statDist(i, list_total_diseases)
        if distance < t:#TODO
            pass



def statDist(l1, l2):
    return 1

if __name__ == '__main__':
    original_df = pd.read_csv('dataset-Privacy-Engineering.csv')

    encoded_df = features_encoding(original_df)
    hashed_df = remove_direct_identifier(encoded_df)
    pseudonimized_df = generalize_data(hashed_df)
    anonymised_df = k_anonymization(pseudonimized_df, k=10)
    anonymised_df = anonymised_df.drop(["gender", "number_vehicles", "education", "children", "marital_status",
                                        "employment", "commute_time", "accommodation"], axis=1)
    diversified_df, _ = l_diversity(anonymised_df, l=8)
    t_close_df = t_closeness(diversified_df)
    print(f"Loss between original and pseudominized : {loss_entropy(original_df, pseudonimized_df)}")
    print(f"Loss between pseudominized and anonymized : {loss_entropy(pseudonimized_df, anonymised_df)}")
    print(f"Loss between anonymized and divsersified : {loss_entropy(anonymised_df, diversified_df)}")


    anonymised_df.to_csv('test2.csv')
