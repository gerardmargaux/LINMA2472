#######################################################################################################################
#                                           CODE FOR ASSIGNMENT 1                                                     #
#######################################################################################################################
import itertools

import community
import gensim
from matplotlib import cm
from nltk.cluster import KMeansClusterer
from gensim.parsing.preprocessing import preprocess_string, strip_short
from gensim.parsing.preprocessing import strip_tags       # strip html tags
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum
import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

listOfName = ["Pavlovna", "Vasili Kuragin", "Helene", "Pierre", "Hippolyte", "Zherkov", "Captain Timokhin", "Alpatych", "Weyrother",
              "Mortemart", "Morio", "Bolkonskaya", "Nicholas", "Joseph Alexeevich", "Peronskaya", "Vasili Dmitrich", "Tikhon", "Dolgorukov", "Langeron",
              "Mikhaylovna", "Anatole Kuragin",  "Dolokhov", "Stevens",  "Countess Rostova", "Denisov", "Likhachev","Lavrushka", "Miloradovich", "Emperor",
              "Count Ilya Rostov", "Count Rostov", "Natasha", "Petya",  "Sonya", "Drubetskoy", "Bonaparte", "Kirsten",
              "Dmitri", "Marya Lvovna Karagina", "Karagina", "Count Cyril", "Bilibin", "Repnin","Bourienne",
              "Shinshin", "Jacquot", "Marya Dmitrievna", "Kuzmich", "Marya Fedorovna", "Anisya Fedorovna"]


def build_dict_persons(filename):
    dictForNames = {}
    with open(filename) as f:
        while True:
            line = f.readline()
            if line != "":
                # retire les caracteres spéciaux
                line = castLine(line)
                for name in listOfName:
                    # La moitie des apparition sont sous le nom Count Ilya Rostov, l'autre moitie Count Rostov
                    if name == "Count Ilya Rostov":
                        name = name.replace("Ilya", "")
                    if name.lower() in line.lower():
                        if name.lower() in dictForNames:
                            dictForNames[name.lower()] += 1
                        else:
                            dictForNames[name.lower()] = 1
            else:
                return dictForNames


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


def preprocessing_as_token(sentences, dict_pers):
    all_sentences = sentences
    # Preprocessing filters
    filters = [lambda x: strip_short(x.lower(), 4), strip_multiple_whitespaces, strip_tags, strip_punctuation,
               strip_non_alphanum]
    all_sentences_preprocessed = []
    for sent in all_sentences:
        parsed_line = preprocess_string(sent, filters)
        all_sentences_preprocessed.append(parsed_line)
    return all_sentences_preprocessed


def word2vec(filename):
    compound_person_dict = {}
    for person, count in build_dict_persons(filename).items():
        person_list = person.split()
        if len(person_list) > 1:
            compound_person_dict[person] = "_".join(person_list)
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

    preprocessed_sentences = preprocessing_as_token(all_sentences_clean2, build_dict_persons(filename))

    # Define the model
    model = gensim.models.Word2Vec(size=300, window=30, min_count=1, alpha=0.1)
    #model = gensim.models.Word2Vec(window=wind, min_count=1)
    model.build_vocab(preprocessed_sentences)

    # Train the model
    model.train(preprocessed_sentences, total_examples=model.corpus_count, compute_loss=True, epochs=100)

    # Get vocabulary from the model
    vocabulary = list(model.wv.vocab)

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
                 fontsize=10, alpha=0.5)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

    return person_embedding_vectors, V_tranform, person_set_vocab


def clustering(embedding_vectors, V_transform, vocabulary, n_clusters=5):
    NUM_CLUSTERS = n_clusters
    kclusterer_sklearn = KMeans(n_clusters=NUM_CLUSTERS)

    # be careful to supply the projected vectors (2D) to the algorithm!
    assigned_clusters = kclusterer_sklearn.fit_predict(V_tranform)

    # DEFINE COLORS OF CLUSTERS
    colors = []
    for lab in assigned_clusters:
        if lab == 0:
            colors.append("cyan")
        elif lab == 1:
            colors.append("magenta")
        elif lab == 2:
            colors.append("green")
        elif lab == 3:
            colors.append("orange")
        elif lab == 4:
            colors.append("red")
        else:
            colors.append("purple")

    # PLOT THE RESULTS
    fig = plt.figure(figsize=(10, 8), dpi=90)
    plt.scatter(*zip(*V_tranform), marker='.', s=100, lw=0, alpha=0.7, c=colors,
                edgecolor='k')

    for i, (x, y) in enumerate(V_tranform):
        plt.text(x, y, vocabulary[i], color=colors[i],
                 fontsize=10, alpha=0.7)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
    return assigned_clusters


def addToGraph(G, listName):
    """
    An edge was added in G for all pairs of names in listName
    :param G: represents the graph
    :param listName: represents the list of all characters
    """
    for i in range(len(listName)):
        for j in range(i+1, len(listName)):
            G.add_edge(listName[i], listName[j])


def buildGraph2():
    """
    Builds the graph and links 2 characters if they appear in the same paragraph
    :return : The builded graph
    """
    dictForNames = {}
    listOfNameInParagraph = []
    G = nx.Graph()
    with open("./book.txt") as f:
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
                        if name.lower() in dictForNames:
                            dictForNames[name.lower()] += 1
                        else:
                            dictForNames[name.lower()] = 1
                            listOfNameInParagraph.append(name.lower())
                            G.add_node(name.lower())
            else:
                return G


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def louvain_algo():
    G = buildGraph2()
    partitions = {}
    part = community.best_partition(G)
    for k, v in part.items():
        if k in partitions:
            partitions[k].append(v)
        else:
            partitions[k] = [v]

    partition_sorted = dict(sorted(partitions.items()))
    maxPartition = 0
    assigned_clusters = []
    for key, val in partition_sorted.items():
        maxPartition = max(partition_sorted[key][0], maxPartition)
        assigned_clusters.append(val[0])

    colors = ["cyan", "magenta", "green", "orange", "red", "purple"]
    pos = community_layout(G, part)
    labels = {}
    for node in G.nodes():
        labels[node] = node
    for com in set(part.values()):
        list_nodes = [nodes for nodes in part.keys() if part[nodes] == com]
        labs = {k:k for k in list_nodes}
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20, node_color=colors[com])
        nx.draw_networkx_labels(G, pos, labs, font_size=9, font_color=colors[com], verticalalignment='bottom',
                                horizontalalignment='left')
    plt.show()
    return partition_sorted, maxPartition


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
    if len(labels1) != len(labels2):
        raise Exception()
    for i, j in itertools.combinations(range(n), 2):
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
    book = './book.txt'
    partitionLouvain, n_clusters = louvain_algo()
    embedding_vectors, V_tranform, vocabulary = word2vec(book)
    assigned_clusters = clustering(embedding_vectors, V_tranform, vocabulary, n_clusters+1)
    result = jaccard_similarity(assigned_clusters, list(partitionLouvain.values()))
    print(result)

