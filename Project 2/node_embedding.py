#######################################################################################################################
#                                           CODE FOR ASSIGNMENT 1                                                     #
#######################################################################################################################
import itertools
import gensim
from matplotlib import cm
from nltk.cluster import KMeansClusterer
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
import networkx as nx
import community
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib.numpy_pickle_utils import xrange

listOfName = ["Pavlovna", "Vasili Kuragin", "Helene", "Pierre", "Hippolyte", "Zherkov", "Captain Timokhin", "Alpatych", "Weyrother",
              "Mortemart", "Morio", "Bolkonskaya", "Nicholas", "Joseph Alexeevich", "Peronskaya", "Vasili Dmitrich", "Tikhon", "Dolgorukov", "Langeron",
              "Mikhaylovna", "Anatole Kuragin",  "Dolokhov", "Stevens",  "Countess Rostova", "Denisov", "Likhachev","Lavrushka", "Miloradovich", "Emperor",
              "Count Ilya Rostov", "Count Rostov", "Natasha", "Petya",  "Sonya", "Drubetskoy", "Bonaparte", "Kirsten",
              "Dmitri", "Marya Lvovna Karagina", "Karagina", "Count Cyril", "Bilibin", "Repnin","Bourienne",
              "Shinshin", "Jacquot", "Marya Dmitrievna", "Kuzmich", "Marya Fedorovna", "Anisya Fedorovna"]


def build_dict_persons(filename):
    dictForNames = {}
    dictForNamesTot = {}
    listOfNameInParagraph = []
    G = nx.Graph()
    with open(filename) as f:
        while True:
            line = f.readline()
            if line == "\n":
                addToGraph(G, listOfNameInParagraph)
                dictForNames = {}
                listOfNameInParagraph = []
            elif line != "":
                # retire les caracteres spéciaux
                line = castLine(line)
                for name in listOfName:
                    # La moitie des apparition sont sous le nom Count Ilya Rostov, l'autre moitie Count Rostov
                    if name == "Count Ilya Rostov":
                        name = name.replace("Ilya", "")
                    if name.lower() in line.lower():
                        dictForNames[name.lower()] = dictForNames.get(name.lower(), 0) + 1
                        dictForNamesTot[name.lower()] = dictForNamesTot.get(name.lower(), 0) + 1
                        if dictForNames[name.lower()] == 1:
                            listOfNameInParagraph.append(name.lower())
                            G.add_node(name.lower())
            else:
                return dictForNamesTot, G


def castLine(line):
    """
    :param line: The line of the file that we are reading
    :return:  The accent were replaced by the same letter without accent
    """
    line = line.replace("é", "e")
    line = line.replace("ë", "e")
    line = line.replace("è", "e")
    line = line.replace("à", "a")
    line = line.replace("á", "a")
    line = line.replace("ó", "o")
    line = line.replace("í", "i")
    line = line.replace("ú", "u")
    line = line.replace(",", "")
    return line

def addToGraph(G, listName):
    """
    An edge was added in G for all pairs of names in listName
    :param G: represents the graph
    :param listName: represents the list of all characters
    """
    for i in range(len(listName)):
        for j in range(i+1, len(listName)):
            G.add_edge(listName[i], listName[j])


def preprocessing_as_token(sentences, dict_pers):
    all_sentences = sentences
    # Preprocessing filters
    filters = [lambda x: x.lower(), stem_text, remove_stopwords, strip_multiple_whitespaces, strip_non_alphanum,
               strip_punctuation]
    all_sentences_preprocessed = []
    for sent in all_sentences:
        parsed_line = preprocess_string(sent, filters)
        all_sentences_preprocessed.append(parsed_line)
    return all_sentences_preprocessed


def word2vec(filename, dictPerson, show=True):
    compound_person_dict = {}
    for person, count in dictPerson.items():
        person_list = person.split()
        if len(person_list) > 1:
            compound_person_dict[person] = "".join(person_list)
        else:
            compound_person_dict[person] = person

    sent_detector = nltk.data.load('punkt/english.pickle')
    all_sentences = []
    with open(filename, "r", encoding="utf-8") as fp:
        text = fp.read()
        text = text.replace("\n\n", " ")
        text = castLine(text)
        all_sentences = sent_detector.tokenize(text)

    all_sentences_clean2 = []
    for sent in all_sentences:
        sent_clean = sent.lower()
        for person, compound_person in compound_person_dict.items():
            sent_clean = sent_clean.replace(person, compound_person)
        all_sentences_clean2.append(sent_clean)

    preprocessed_sentences = preprocessing_as_token(all_sentences_clean2, dictPerson)

    # Define the model
    model = gensim.models.Word2Vec(size=5, window=10, min_count=1, alpha=0.01)
    model.build_vocab(preprocessed_sentences)

    # Train the model
    model.train(preprocessed_sentences, total_examples=model.corpus_count, compute_loss=True, epochs=100)

    # Get vocabulary from the model
    vocabulary = list(model.wv.vocab)
    print(f"Vocabulary size :: {len(vocabulary)}")

    person_set_processed = [s for s in compound_person_dict.values() if s.lower() in vocabulary]
    person_set_vocab = [s.lower() for s in person_set_processed]

    # Get word embedding vectors
    person_embedding_vectors = model[person_set_vocab]

    # PROJECT IT INTO 2D
    V_tranform = TSNE(n_components=2).fit_transform(person_embedding_vectors)

    # PLOT THE PROJECTION
    fig = plt.figure(figsize=(10, 8), dpi=90)
    plt.scatter(*zip(*V_tranform), marker='.', s=100, lw=0, alpha=0.7,
                edgecolor='k')
    for i, (x, y) in enumerate(V_tranform):
        plt.text(x, y, person_set_vocab[i],
                 fontsize=12, alpha=0.5)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

    return person_embedding_vectors, V_tranform, person_set_vocab


def clustering(embedding_vectors, V_transform, vocabulary, NUM_CLUSTERS = 4):
    kclusterer_sklearn = KMeans(n_clusters=NUM_CLUSTERS)

    # be careful to supply the projected vectors (2D) to the algorithm!
    assigned_clusters = kclusterer_sklearn.fit_predict(V_transform)
    # DEFINE COLORS OF CLUSTERS
    colors = cm.nipy_spectral(np.array(assigned_clusters).astype(float) / NUM_CLUSTERS)

    # PLOT THE RESULTS
    fig = plt.figure(figsize=(10, 8), dpi=90)
    plt.scatter(*zip(*V_transform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
                edgecolor='k')

    for i, (x, y) in enumerate(V_transform):
        plt.text(x, y, vocabulary[i], color=colors[i],
                 fontsize=12, alpha=0.5)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
    return list(zip(vocabulary, assigned_clusters))


def jaccard_similarity(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(xrange(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)


if __name__ == '__main__':
    book = '../Project 1/book.txt'
    dict_pers, G = build_dict_persons(book)
    #print(dict_pers)
    #sentences = preprocessing_as_token(book)
    embedding_vectors, V_tranform, person_set_vocab = word2vec(book, dict_pers, show=False)
    partitionLouvain = community.best_partition(G)
    maxPartition = 0
    for i in partitionLouvain:
        maxPartition = max(partitionLouvain[i], maxPartition)
    print(maxPartition+1)
    cluster = clustering(embedding_vectors, V_tranform, person_set_vocab, maxPartition+1)
    sortedLouvain = sorted(partitionLouvain.items())
    sortedLouvain = sortedLouvain[:9] + [sortedLouvain[10]] + [sortedLouvain[9]] + sortedLouvain[11:]
    sortedCluster = sorted(cluster)
    print(jaccard_similarity([n for (_, n) in sortedLouvain], [n for (_, n) in sortedCluster]))

    #print(jaccard_similarity(cluster, partitionLouvain))
