import hashlib
import math
from copy import deepcopy

import pandas as pd
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
    new_df['dob'] = [list_dob[i].split("/")[1] for i in range(len(list_dob))]

    list_zipcode = new_df['zipcode'].tolist()
    new_df['zipcode'] = [str(list_zipcode[i])[0:2]+str('***') for i in range(len(list_zipcode))]
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


def form_classes(df):
    sub_df = df[["gender", "dob", "zipcode"]]
    list_tuples = sub_df.to_records()  # Rows of the sub-dataframe into tuples
    dict_classes = dict()
    for (index, gender, dob, zipcode) in list_tuples:
        if (gender, dob, zipcode) in dict_classes:
            dict_classes[(gender, dob, zipcode)] += 1
        else:
            dict_classes[(gender, dob, zipcode)] = 1

    dict_classes = dict(sorted(dict_classes.items(), key=lambda x: x[1], reverse=False))
    return dict_classes


def k_anonymization(df, dict_classes, k=2):
    """
    Provides a k-anonymized data frame where unique tuples were removed from the data frame
    :param df: input data frame to transform
    :return: k-anonymized data frame
    """
    index_to_drop = []
    for key, val in dict_classes.items():
        if val >= k:
            break
        else:
            index = df.index[(df["gender"] == key[0]) & (df["dob"] == key[1]) & (df["zipcode"] == key[2])].tolist()
            index_to_drop.append(index[0])

    df = df.drop(index_to_drop, axis=0)

    return df


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


if __name__ == '__main__':
    original_df = pd.read_csv('dataset-Privacy-Engineering.csv')

    encoded_df = features_encoding(original_df)
    hashed_df = remove_direct_identifier(encoded_df)
    pseudonimized_df = generalize_data(hashed_df)
    dict_classes = form_classes(pseudonimized_df)
    anonymised_df = k_anonymization(pseudonimized_df, dict_classes, k=3)

    print(f"Loss between original and pseudominized : {loss_entropy(original_df, pseudonimized_df)}")
    print(f"Loss between pseudominized and anonymized : {loss_entropy(pseudonimized_df, anonymised_df)}")

    #df.to_csv('test.csv')
