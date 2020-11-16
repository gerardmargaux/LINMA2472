import hashlib

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def remove_direct_identifier(df):
    """
    Replaces the unique direct identifier (id) by a hashed and salted id
    :param df: input data frame to transform
    :return: data frame where the id feature was replaced by hashed and salted id
    """
    list_id = df['id'].tolist()
    new_list_id = []
    for name in list_id:
        name_to_hash = "MargauxGerard" + name.replace(" ", '') + "JeremieBogaert"  # Salting
        new_list_id.append(name_to_hash)
    df['id'] = [hashlib.sha1(str.encode(str(i))).hexdigest() for i in new_list_id]  # Hashing
    return df


def generalize_date_of_birth(df):
    """
    Transforms the date of birth of a person by only the year of the birth
    :param df: input data frame to transform
    :return: data frame where the dob feature was replaced by the year of birth
    """
    list_dob = df['dob'].tolist()
    df['dob'] = [list_dob[i].split("/")[1] for i in range(len(list_dob))]
    return df


def features_encoding(df):
    """
    Encodes some features as numerical values. For example, we will encode gender as : female -> 0, male -> 1
    :param df: input data frame to encode
    :return: data frame where some features were encoded
    """
    encoder = LabelEncoder()
    to_be_encoded = ["gender", "education", "employment", "marital_status", "ancestry", "accommodation"]
    for feature in to_be_encoded:
        df[feature] = encoder.fit_transform(df[feature])
    return df


if __name__ == '__main__':
    original_df = pd.read_csv('dataset-Privacy-Engineering.csv')
    encoded_df = features_encoding(original_df)
    hashed_df = remove_direct_identifier(encoded_df)
    pseudonimized_df = generalize_date_of_birth(hashed_df)
    #print(pseudonimized_df)
