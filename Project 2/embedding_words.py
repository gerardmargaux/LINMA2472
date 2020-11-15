#######################################################################################################################
#                                           CODE FOR ASSIGNMENT 2                                                     #
#######################################################################################################################
import re
import string
from copy import deepcopy

import gensim
import pandas as pd
import sklearn
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn import preprocessing
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags       # strip html tags
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum
from sentence_transformers import SentenceTransformer
import transformers as ppb
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, LSTM, MaxPooling1D, Flatten
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from transformers import XLNetTokenizer, TFBertForSequenceClassification, AutoTokenizer, AutoModel


def apply_filters(text):
    # Preprocessing filters
    result = []
    filters = [lambda x: x.lower(), stem_text, strip_punctuation, strip_non_alphanum, remove_stopwords,
               strip_multiple_whitespaces]
    for sent in text:
        parsed_line = preprocess_string(sent, filters)
        result.append(parsed_line)
    return sum(result, [])


def before_proc(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


def preprocessing_fun(df, token=True):
    sent_detector = nltk.data.load('punkt/english.pickle')
    if token:
        df['preprocessed'] = df.apply(lambda row: sent_detector.tokenize(row['body']), axis=1).apply(apply_filters)
    else:
        df['preprocessed'] = df['body'].apply(lambda row: before_proc(row))
    return df


def bert(df_train, long=False):
    tqdm.pandas()
    if long:
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        encoded = df_train['preprocessed'].progress_apply(lambda row: embedder.encode(row))
        final_df = pd.DataFrame({'col': list(encoded)})
        final_df.to_csv('bert.csv')

    else:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        df_train = preprocessing_fun(df_train, token=False)
        listOfEncoding = model.encode(df_train['preprocessed'].tolist(), show_progress_bar=True)
        lastLen = len(listOfEncoding[0])
        for i in listOfEncoding:
            if lastLen != len(i):
                print("Different length :", lastLen, len)
            lastLen = len(i)
        final_df = pd.DataFrame(listOfEncoding)
        final_df.reset_index(inplace=True)
        df_train.reset_index(inplace=True)
        final_df['subreddit'] = df_train['subreddit']

        print('finished Embedding')
        final_df.to_csv('bert.csv')

    return final_df


def doc2vec_save(df_train):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_train['preprocessed'].tolist())]

    # Define the model
    doc_model = gensim.models.Doc2Vec(vector_size=300, window=4, min_count=4, alpha=0.01)
    doc_model.build_vocab(documents)

    # Train the model
    max_epochs = 20
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        doc_model.train(documents, total_examples=doc_model.corpus_count, epochs=doc_model.iter)
        # increase the learning rate
        doc_model.alpha -= 0.0002
        # fix the learning rate
        doc_model.min_alpha = doc_model.alpha

    return doc_model


def doc2vec_load(df_test, df_train, nbr_test, model=None):
    if model is None:
        doc_model = Doc2Vec.load("doc2vec.model")
    else:
        doc_model = model
    dv = doc_model.docvecs.vectors_docs

    final_df_tr = pd.DataFrame(dv[:-nbr_test])
    final_df_tr['subreddit'] = df_train['subreddit'].tolist()

    final_df_te = pd.DataFrame(dv[-nbr_test:])
    final_df_te['subreddit'] = df_test['subreddit'].tolist()

    return final_df_tr, final_df_te


def classifier(df_train, df_test, bert_embedding=True):

    if bert_embedding:  # With bert embedding
        df_train = preprocessing_fun(df_train, token=False)
        df_test = preprocessing_fun(df_test, token=False)
        nbr_test = len(df_test)
        #df_tot = df_train.append(df_test)
        #df_tot = bert(df_tot, long=False)
        df_tot = pd.read_csv('bert.csv')
        df_train_encoded = df_tot.iloc[:-nbr_test]
        df_test_encoded = df_tot.iloc[-nbr_test:]

    else:  # With Doc2Vec embedding
        df_train = preprocessing_fun(df_train, token=True)
        df_test = preprocessing_fun(df_test, token=True)
        nbr_test = len(df_test)
        df_tot = df_train.append(df_test)
        model = doc2vec_save(df_tot)
        df_train_encoded, df_test_encoded = doc2vec_load(df_test, df_train, nbr_test, model=model)

    min_max_scaler = preprocessing.MinMaxScaler()  # Normalization of data between 0 and 1
    X_train = df_train_encoded.loc[:, df_train_encoded.columns != 'subreddit'].to_numpy()
    X_train = min_max_scaler.fit_transform(X_train)
    y_train = df_train_encoded['subreddit']
    X_test = df_test_encoded.loc[:, df_train_encoded.columns != 'subreddit'].to_numpy()
    X_test = min_max_scaler.fit_transform(X_test)
    y_test = df_test_encoded['subreddit']
    X_train_keras = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_keras = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Neural network model
    model = tf.keras.Sequential()
    model.add(Conv1D(128, 3, input_shape=X_train_keras.shape[1:3], activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_keras, y_train, batch_size=32, epochs=15, shuffle=True)
    eval = model.evaluate(X_test_keras, y_test)
    print("Keras model test: %.2f%%" % (eval[1] * 100))
    eval = model.evaluate(X_train_keras, y_train)
    print("Keras model train: %.2f%%" % (eval[1] * 100))

    # Classification and prediction
    models = {"KNN": KNeighborsClassifier(n_neighbors=10, weights='distance'),
              "RandomForest": RandomForestClassifier(max_depth=40, random_state=2),
              "Logistic": LogisticRegression(max_iter=2000),
              "NaiveBayes": MultinomialNB(),
              "GradientBoosting": GradientBoostingClassifier(n_estimators=10, max_depth=20)}

    for name, model in models.items():
        classifier = model.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred2 = classifier.predict(X_train)
        acc = accuracy_score(y_test, y_pred)*100
        acc2 = accuracy_score(y_train, y_pred2)*100
        print(f"{name} test : {acc} %")
        print(f"{name} train : {acc2} %")

    return


if __name__ == '__main__':
    df_biden_train = pd.read_csv('./comments/JoeBiden_train.csv').rename(columns={"Unnamed: 0": "index"}).set_index("index")
    df_biden_train['subreddit'] = 0
    df_biden_test = pd.read_csv('./comments/JoeBiden_test.csv').rename(columns={"Unnamed: 0": "index"}).set_index("index")
    df_biden_test['subreddit'] = 0
    df_trump_train = pd.read_csv('./comments/The_Donald_train.csv').rename(columns={"Unnamed: 0": "index"}).set_index("index")
    df_trump_train['subreddit'] = 1
    df_trump_test = pd.read_csv('./comments/The_Donald_test.csv').rename(columns={"Unnamed: 0": "index"}).set_index("index")
    df_trump_test['subreddit'] = 1

    df_test = pd.concat([df_biden_test, df_trump_test], axis=0)
    df_train = pd.concat([df_biden_train, df_trump_train], axis=0)
    df_test = df_test.sample(frac=1).reset_index().drop(columns="index")
    df_train = df_train.sample(frac=1).reset_index().drop(columns="index")
    classifier(df_train, df_test)  # With Bert embedding
    #classifier(df_train, df_test, bert_embedding=False)  # With Doc2Vec embedding
