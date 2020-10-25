import math
import sys
import subprocess
# implement pip as a subprocess:
from math import e
from random import choice, sample

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-louvain'])

import community
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from collections import Counter

import networkx as nx
import pandas as pd


#Liste non exhaustive des personnages, 52 personnages identifiés
listOfName = ["Pavlovna", "Vasili Kuragin", "Helene", "Pierre", "Hippolyte", "Zherkov", "Captain Timokhin", "Alpatych", "Weyrother",
              "Mortemart", "Morio", "Bolkonskaya", "Nicholas", "Joseph Alexeevich", "Peronskaya", "Vasili Dmitrich", "Tikhon", "Dolgorukov", "Langeron",
              "Mikhaylovna", "Anatole Kuragin",  "Dolokhov", "Stevens" ,  "Countess Rostova", "Denisov", "Likhachev","Lavrushka", "Miloradovich", "The Emperor",
              "Count Ilya Rostov", "Count Rostov",  "Vera Rostova", "Nikolai Rostov",  "Natasha", "Petya",  "Sonya", "Drubetskoy", "Bonaparte", "Kirsten",
              "Dmitri", "Marya Lvovna Karagina", "Karagina", "Count Cyril", "Princess Katerina Mamontova", "Bilibin","Captain Tushin", "Repnin","Bourienne",
              "Shinshin", "Julie Drubetskaya", "Jacquot", "Márya Dmítrievna", "Kuzmich", "Marya Fedorovna", "Anisya Fedorovna"]


def buildGraph():
    """
    Builds the graph and links 2 characters if they appear in the same paragraph
    :return : The builded graph
    """
    dictForNames = {}
    listOfNameInParagraph = []
    G = nx.Graph()
    with open("book.txt") as f:
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
                        name.replace("Ilya", "")
                    if name.lower() in line.lower():
                        if name.lower() in dictForNames:
                            dictForNames[name.lower()] += 1
                        else:
                            dictForNames[name.lower()] = 1
                            listOfNameInParagraph.append(name.lower())
                            G.add_node(name.lower())
            else:
                return G


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


def louvain_algorithm(G):
    # Source : https://perso.crans.org/aynaud/communities/
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    print("Degree assortativity = ", degree_assortativity)
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=40, node_color=str(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    return


def k_cores_decomposition(G):
    """
    Computes the k-cores of G
    :param G: represents a graph
    :return: the k-cores of G in a dictionary containing {k, nodes_in_k_core}
    """
    dictOfDegree = {}
    for i in G.nodes():
        dictOfDegree[i] = [G.degree[i], False]
    k = 0
    pruned = 0
    dictKcore = {}
    while pruned < G.number_of_nodes():
        to_prune = list([i for i in dictOfDegree.keys() if dictOfDegree[i][0] == k])
        dictKcore[k] = []
        while len(to_prune) > 0:
            nextToPrune = to_prune.pop()
            for neighbors in G.adj[nextToPrune]:
                if not dictOfDegree[neighbors][1]:
                    dictOfDegree[neighbors][0] -= 1
                    if dictOfDegree[neighbors][0] == k:
                        to_prune.append(neighbors)
            dictKcore[k].append(nextToPrune)
            dictOfDegree[nextToPrune][1] = True
            pruned += 1
        k += 1
    return dictKcore


def barabasi_albert_generation(G):
    """
    Generates a random graph thanks to the barabasi-albert algorithm
    :return: The generated graph
    """
    average_degree = 0
    for node in G.nodes():
        average_degree += G.degree(node)

    average_degree /= len(listOfName)
    barabasi_graph = nx.barabasi_albert_graph(n=len(listOfName), m=round(average_degree), seed=1998)
    mapping = {}
    for i in range(len(listOfName)):
        mapping[i] = listOfName[i]

    barabasi_graph = nx.relabel_nodes(barabasi_graph, mapping)
    degree_assortativity = nx.degree_assortativity_coefficient(barabasi_graph)
    print("Degree assortativity = ", degree_assortativity)

    # k-cores decomposition of barabasi-albert algo
    kCores = k_cores_decomposition(barabasi_graph)
    for a, b in kCores.items():
        print(a, "core :", b)

    # Comparison of G and barabasi-albert
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.flatten()
    nx.draw_networkx(barabasi_graph, ax=ax[0])
    ax[0].set_axis_off()
    ax[0].title.set_text('Generation of Barabasi-Albert')
    louvain_algorithm(barabasi_graph)
    ax[1].title.set_text('Louvain algorithm of Barabasi-Albert')
    ax[1].set_axis_off()
    plt.show()

    """fig2, axes2 = plt.subplots(nrows=1, ncols=2)
    ax_graph = axes2.flatten()
    draw_graph(G)
    ax_graph[0].set_axis_off()
    ax_graph[0].title.set_text('Co-occurrence network of characters')
    louvain_algorithm(G)
    ax_graph[1].set_axis_off()
    ax_graph[1].title.set_text('Louvain algorithm of co-occurrence network')
    fig.show()
    fig2.show()"""
    return


def draw_graph(G):
    """
    Plot the network build based on the graph given in argument
    :param G: the graph of the co-occurrence of characters
    """
    mapping = {}
    for i in range(len(listOfName)):
        mapping[i] = listOfName[i]
    G = nx.relabel_nodes(G, mapping)
    nx.draw(G, node_size=30, with_labels=True)
    plt.show()


def independent_cascade(G, p=0.1):
    """
    Defines the set of nodes that are activated in the graph
    :param G: represents the graph
    :param p: is the probability that a node infects its neighbours
    :return: the number of iteration needed to end the algorithm
    """
    count = 0
    while len(new_infected) > 0:
        #print("New infected neighbours :", new_infected)
        #print("Total infected neighbours :", total_infected)
        new_infected_iter = deepcopy(new_infected)
        for node in new_infected_iter:
            if node not in total_infected:
                total_infected.append(node)
            neighbours = list(G.neighbors(node))
            not_infected_yet = []
            for neigh in neighbours:
                if neigh not in total_infected:
                    not_infected_yet.append(neigh)
            val = p * len(not_infected_yet)
            number_of_activated = math.ceil(val)
            random.shuffle(not_infected_yet)
            for n in not_infected_yet[:number_of_activated]:
                total_infected.append(n)
                new_infected.append(n)
            new_infected.remove(node)
        count += 1
    #print("-----------------------------------END--------------------------------------------")

    return count


def create_set(S):
    """
    Shuffles the set S and add the element of this set to total_infected and new_infected
    :param S: Set containing items we want to add
    :return:
    """
    random.shuffle(list(S))
    S = set(S)
    for item in S:
        total_infected.append(item)
        new_infected.append(item)


def inluence_maximization_problem_greedy(G):
    """
    Greedy algorithm of the influence maximization problem
    :param G: Graph used for determining the link between characters
    :return: The final set containing 5% of the total number of nodes that maximizes the influence in the network
    """
    nodes_degree = dict()  # Dict of {node:degree}
    S = []  # Initial set of nodes
    nodes_in_set = math.ceil(G.number_of_nodes() * 0.05)
    for node in G.nodes():
        nodes_degree[node] = G.degree(node)

    nodes_degree = dict(sorted(nodes_degree.items(), key=lambda x: x[1], reverse=True))

    for i in range(nodes_in_set):
        if len(S) == 0:
            first_key = list(nodes_degree.keys())[0]
            S.append(first_key)  # Adding the node with highest degree
            del nodes_degree[first_key]
        else:
            neighbours = tuple()  # Tuple containing all neighbours of nodes present in S
            for node in S:
                neighbours = neighbours + tuple(G.neighbors(node))

            for key, val in nodes_degree.items():
                if key not in S and key in neighbours:
                    S.append(key)
                    break
    return S


def influence_maximization_problem(G):
    """
    Algorithm of the influence maximization problem with hill-climbing heuristic
    :param G: Graph used for determining the link between characters
    :return: The final set containing 5% of the total number of nodes that maximizes the influence in the network
    """
    S = set()  # Initial set of nodes
    R = 1  # Number of random cascades
    nodes_in_set = math.ceil(G.number_of_nodes()*0.05)
    for i in range(nodes_in_set):
        best_node = None
        best_sv = 0
        for node in G.nodes():
            a = node
            if node not in S:
                s_v = 0
                new_set = S | {node}
                for j in range(R):
                    total_infected.clear()
                    create_set(new_set)
                    iter = independent_cascade(G)
                    s_v += len(total_infected)
                s_v /= R
                if s_v > best_sv:
                    best_sv = s_v
                    best_node = node
        S = S | {best_node}

    return S

if "__main__":
    G = buildGraph()
    #louvain_algorithm(G)
    #kCores = k_cores_decomposition(G)
    #for a, b in kCores.items():
    #    print(a, "core :", b)
    barabasi_albert_generation(G)
    total_infected = []
    new_infected = []
    #count = independent_cascade(G)
    #print("Number of iterations = ", count)
    #S = influence_maximization_problem(G)
    #S = inluence_maximization_problem_greedy(G)
    #print("Final set : ", S)
