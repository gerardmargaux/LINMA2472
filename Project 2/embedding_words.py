#######################################################################################################################
#                                           CODE FOR ASSIGNMENT 2                                                     #
#######################################################################################################################
import gensim
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags       # strip html tags
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum
from sentence_transformers import SentenceTransformer
from sklearn.datasets import make_classification
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#import subprocess
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentence-transformers'])


def preprocessing_as_token(filename):
    sent_detector = nltk.data.load('punkt/english.pickle')
    all_sentences = []
    with open(filename, "r", encoding="utf-8") as fp:
        text = fp.read()
        text = text.replace("\n\n", " ")
        all_sentences = sent_detector.tokenize(text)
    # Preprocessing filters
    filters = [lambda x: x.lower(), stem_text, strip_punctuation, strip_non_alphanum, remove_stopwords,
               strip_multiple_whitespaces]
    all_sentences_preprocessed = []
    for sent in all_sentences:
        parsed_line = preprocess_string(sent, filters)
        all_sentences_preprocessed.append(parsed_line)
    return all_sentences_preprocessed


def bert(filename):
    sent_detector = nltk.data.load('punkt/english.pickle')
    embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

    sentences = []
    with open(filename, "r", encoding="utf-8") as fp:
        text = fp.read()
        text = text.replace("\n\n", " ")
        sentences = sent_detector.tokenize(text)
    encoded_sentences = embedder.encode(sentences, show_progress_bar=True)

    # Classification
    X, y = make_classification(n_samples=4000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=True)

    # Split train and test set -> Train 80%/ Test 20%
    threshold = int(0.8 * len(X))
    X_train, X_test = X[:threshold], X[threshold:]
    y_train, y_test = y[:threshold], y[threshold:]

    # Training of the classifier
    classifierRF = RandomForestClassifier(max_depth=5, random_state=0)
    classifierRF.fit(X_train, y_train)

    # Prediction
    y_pred = classifierRF.predict(X_test)
    y_train_pred = classifierRF.predict(X_train)

    # Compute accuracy
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    print(f"Accuracy:\n  training :: {acc_train}\n  test     :: {acc_test}")


def word2vec(all_sentences_preprocessed):
    # Define the model
    model = gensim.models.Word2Vec(size=100, window=4, min_count=10, alpha=0.01)
    model.build_vocab(all_sentences_preprocessed)

    iterations = range(10)
    # Train the model
    for i in iterations:
        model.train(all_sentences_preprocessed, total_examples=model.corpus_count, compute_loss=True, epochs=100)

    # Get vocabulary from the model
    vocabulary = list(model.wv.vocab)
    print(f"Vocabulary size :: {len(vocabulary)}")

    # Get word embedding vectors
    embedding_vectors = model[model.wv.vocab]

    # Transform embedding vectors into 2D and plot the result
    V_tranform = TSNE(n_components=2).fit_transform(embedding_vectors)
    fig = plt.figure(figsize=(10, 8), dpi=90)
    plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7, edgecolor='k')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


if __name__ == '__main__':
    path_biden_train = './comments/JoeBiden_train.csv'
    path_biden_test = './comments/JoeBiden_test.csv'
    path_trump_train = './comments/The_Donald_train.csv'
    path_trump_test = './comments/The_Donald_test.csv'

    train_corpus = preprocessing_as_token(path_biden_train)
    test_corpus = preprocessing_as_token(path_biden_test)
    bert(path_biden_train)
