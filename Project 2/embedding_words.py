#######################################################################################################################
#                                           CODE FOR ASSIGNMENT 2                                                     #
#######################################################################################################################
import gensim
import pandas as pd
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
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def apply_filters(text):
    # Preprocessing filters
    result = []
    filters = [lambda x: x.lower(), stem_text, strip_punctuation, strip_non_alphanum, remove_stopwords,
               strip_multiple_whitespaces]
    for sent in text:
        parsed_line = preprocess_string(sent, filters)
        result.append(parsed_line)
    return sum(result, [])


def preprocessing_fun(df, token=True):
    sent_detector = nltk.data.load('punkt/english.pickle')
    if token:
        df['preprocessed'] = df.apply(lambda row: sent_detector.tokenize(row['body']), axis=1).apply(apply_filters)
    else:
        df['preprocessed'] = df.apply(lambda row: sent_detector.tokenize(row['body']), axis=1)
    return df


def bert(df_train, long=False):
    if long:
        tqdm.pandas()
        sent_detector = nltk.data.load('punkt/english.pickle')
        embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        encoded = df_train['preprocessed'].progress_apply(lambda row: embedder.encode(row))
        max_len = 0
        for i in encoded.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (max_len - len(i)) for i in encoded.values])
        final_df = pd.DataFrame(padded)
        final_df['subreddit'] = df_train['subreddit'].tolist()

    else:
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = df_train['body'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True,
                                                                       max_length=1000)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        final_df = pd.DataFrame(attention_mask)
        final_df['subreddit'] = df_train['subreddit'].tolist()

    return final_df


def word2vec(df, biden=True):
    # Define the model
    model = gensim.models.Word2Vec(size=100, window=4, min_count=10, alpha=0.01)
    model.build_vocab(df['preprocessed'].tolist())

    # Train the model
    model.train(df['preprocessed'].tolist(), total_examples=model.corpus_count, compute_loss=True, epochs=100)

    # Get vocabulary from the model
    vocabulary = list(model.wv.vocab)
    print(f"Vocabulary size :: {len(vocabulary)}")

    # Get word embedding vectors
    embedding_vectors = model[model.wv.vocab]

    if biden:
        subreddit = [0 for i in range(len(vocabulary))]
    else:
        subreddit = [1 for i in range(len(vocabulary))]

    final_df = pd.DataFrame(embedding_vectors)
    final_df['subreddit'] = subreddit

    return final_df


def classifier(df_biden_train, df_trump_train, df_biden_test, df_trump_test, bert_embedding=True):

    if bert_embedding:  # With bert embedding
        df_test = df_biden_test.append(df_trump_test)
        df_test = preprocessing_fun(df_test, token=True)
        df_test = df_test.sample(frac=1).reset_index().drop(columns="index")

        df_train = pd.concat([df_biden_train, df_trump_train], axis=0)
        df_train = df_train.sample(frac=1).reset_index().drop(columns="index")

        # Preprocessing
        df_train = preprocessing_fun(df_train, token=True)
        nbr_test = len(df_test)
        df_tot = bert(df_train.append(df_test))

        df_train_encoded = df_tot.iloc[:-nbr_test]
        df_test_encoded = df_tot.iloc[-nbr_test:]

    else:  # With Word2Vec embedding
        df_biden_tr = preprocessing_fun(df_biden_train, token=True)
        df_biden_tr = word2vec(df_biden_tr)
        df_trump_tr = preprocessing_fun(df_trump_train, token=True)
        df_trump_tr = word2vec(df_trump_tr, biden=False)
        df_tot_tr = pd.concat([df_biden_tr, df_trump_tr], axis=0)

        df_biden_te = preprocessing_fun(df_biden_test, token=True)
        df_biden_te = word2vec(df_biden_te)
        df_trump_te = preprocessing_fun(df_trump_test, token=True)
        df_trump_te = word2vec(df_trump_te, biden=False)
        df_tot_te = pd.concat([df_biden_te, df_trump_te], axis=0)

        df_train_encoded = df_tot_tr.sample(frac=1).reset_index().drop(columns="index")
        df_test_encoded = df_tot_te.sample(frac=1).reset_index().drop(columns="index")

    min_max_scaler = preprocessing.MinMaxScaler()  # Normalization of data between 0 and 1
    # selector = VarianceThreshold(threshold=0.01)  # Keep only features with a min variance of 0.01
    X_train = df_train_encoded.loc[:, df_train_encoded.columns != 'subreddit']
    X_train = min_max_scaler.fit_transform(X_train.to_numpy())
    y_train = df_train_encoded['subreddit']
    X_test = df_test_encoded.loc[:, df_train_encoded.columns != 'subreddit']
    X_test = min_max_scaler.fit_transform(X_test.to_numpy())
    y_test = df_test_encoded['subreddit']

    # Classification and prediction
    models = {"KNN": KNeighborsClassifier(n_neighbors=15, weights='distance'),
              "RandomForest": RandomForestClassifier(max_depth=40, random_state=1),
              "Logistic": LogisticRegression(max_iter=2000),
              "NaiveBayes": MultinomialNB(),
              "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=40)}

    for name, model in models.items():
        classifier = model.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} : {acc}")


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
    # classifier(df_biden_train, df_trump_train, df_biden_test, df_trump_test)  # With Bert embedding
    classifier(df_biden_train, df_trump_train, df_biden_test, df_trump_test, bert_embedding=False)  # With Word2Vec embedding

