import math
import sys
import subprocess
# implement pip as a subprocess:
from math import e
from random import choice, sample
import numpy as np
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'color'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-Louvain'])
import community
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from collections import Counter

import networkx as nx
import pandas as pd
import color as clr

from colour import Color

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange", "purple"]
#Liste non exhaustive des personnages, 52 personnages identifiés
listOfName = ["Pavlovna", "Vasili Kuragin", "Helene", "Pierre", "Hippolyte", "Zherkov", "Captain Timokhin", "Alpatych", "Weyrother",
              "Mortemart", "Morio", "Bolkonskaya", "Nicholas", "Joseph Alexeevich", "Peronskaya", "Vasili Dmitrich", "Tikhon", "Dolgorukov", "Langeron",
              "Mikhaylovna", "Anatole Kuragin",  "Dolokhov", "Stevens" ,  "Countess Rostova", "Denisov", "Likhachev","Lavrushka", "Miloradovich", "The Emperor",
              "Count Ilya Rostov", "Count Rostov", "Natasha", "Petya",  "Sonya", "Drubetskoy", "Bonaparte", "Kirsten",
              "Dmitri", "Marya Lvovna Karagina", "Karagina", "Count Cyril", "Bilibin", "Repnin","Bourienne",
              "Shinshin", "Jacquot", "Marya Dmitrievna", "Kuzmich", "Marya Fedorovna", "Anisya Fedorovna"]


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
    count = 0
    for com in set(partition.values()):
        count += 1
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=70, node_color=colors[count])
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

    average_degree /= (len(listOfName)*2)
    barabasi_graph = nx.barabasi_albert_graph(n=len(listOfName), m=round(average_degree), seed=1998)
    mapping = {}
    for i in range(len(listOfName)):
        mapping[i] = listOfName[i].lower()

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
    nx.draw_networkx(barabasi_graph, pos = nx.spring_layout(barabasi_graph) ,with_labels = False, node_size = 100, ax=ax[0])
    ax[0].set_axis_off()
    ax[0].title.set_text('Generation of Barabasi-Albert')
    louvain_algorithm(barabasi_graph)
    ax[1].title.set_text('Louvain algorithm of Barabasi-Albert')
    ax[1].set_axis_off()
    plt.show()
    return barabasi_graph


def draw_cores(G, dictKcore):
    nbrCore = len(dictKcore)
    green = Color("green")
    colors = list(green.range_to(Color("red"),nbrCore))
    pos = nx.spring_layout(G)
    for i in range(nbrCore):
        nx.draw_networkx_nodes(G, pos, dictKcore[i], node_size=70, node_color=np.array([colors[i].rgb]))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def independent_cascade(G, base_infected, p=0.1):
    """
    Defines the set of nodes that are activated in the graph
    :param G: represents the graph
    :param p: is the probability that a node infects its neighbours
    :param base_infected: is the list of initial infected characters
    :return: the number of iteration needed to end the algorithm
    """
    count = 0
    new_infected = base_infected
    total_infected = []
    nbr_infected_at_iter = [0]
    while len(new_infected) > 0:

        new_infected_iter = deepcopy(new_infected)
        new_infected = []
        for node in new_infected_iter:
            neighbours = list(G.neighbors(node))
            for neigh in neighbours:
                if neigh not in total_infected and random.random() < p:
                    new_infected.append(neigh)
                    total_infected.append(neigh)
        count += 1
        nbr_infected_at_iter.append(len(total_infected))

    return len(total_infected), nbr_infected_at_iter



def influence_maximization_problem_greedy(G, k):
    """
    Greedy algorithm of the influence maximization problem
    :param G: Graph used for determining the link between characters
    :param k: The percentage of node from the graph that are selected as first infected
    :return: The final set containing k% of the total number of nodes that maximizes the influence in the network
    """
    nodes_degree = {}
    nodes_in_set = math.ceil(G.number_of_nodes() * k)
    for node in G.nodes():
        nodes_degree[node] = G.degree(node)

    nodes_degree = dict(sorted(nodes_degree.items(), key=lambda x: x[1], reverse=True))
    first_key = list(nodes_degree.keys())[0]
    S =  [first_key]# Initial set of nodes
    del nodes_degree[first_key]
    for i in range(nodes_in_set):
        neighbours = []  # Tuple containing all neighbours of nodes present in S
        for node in S:
            neighbours = neighbours + list(G.neighbors(node))
        for key, val in nodes_degree.items():
            if key not in S and key in neighbours:
                S.append(key)
                break
    return S


def influence_maximization_problem(G, k, p=0.1):
    """
    Algorithm of the influence maximization problem with hill-climbing heuristic
    :param G: Graph used for determining the link between characters
    :param k: The percentage of node from the graph that are selected as first infected
    :return: The final set containing k% of the total number of nodes that maximizes the influence in the network
    """
    S = set()  # Initial set of nodes
    R = 100  # Number of random cascades
    nodes_in_set = math.ceil(G.number_of_nodes()*k)
    for i in range(nodes_in_set):
        best_node = None
        best_sv = 0
        for node in G.nodes():
            if node not in S:
                s_v = 0
                new_set = S | {node}
                for j in range(R):
                    total_infected, _ = independent_cascade(G, list(new_set), p)
                    s_v += total_infected
                new_set = S
                s_v /= R
                if s_v > best_sv:
                    best_sv = s_v
                    best_node = node
        S = S | {best_node}
    return list(S)

def generate_Set(G, k, p, opt = 0):
    if opt == 0:
        return sample(list(G.nodes()), math.ceil(k*len(listOfName)))
    elif opt == 1:
        deg = dict()
        for node in G.nodes():
            deg[node] = G.degree(node)
        deg = dict(sorted(deg.items(), key = lambda x: x[1], reverse = True))
        return list(deg.keys())[:3]
    elif opt == 2:
        return influence_maximization_problem(G, k, p)
    else:
        return influence_maximization_problem_greedy(G,k)

def plot_Graph(G, k, p):
    nbrIter = 50
    listModel = ["Random set", "Highest degrees set", "Hill climbing max influence", "Greedy max influence"]
    for i in p:
        for j in k:
            for model in range(4):#0: Random, 1: Highest degree, 2: Hill climbing, 3: Greedy
                listTotInfected = [0 for z in range(30)] #nbr moyen d'infecté a chaque temps
                for iter in range(nbrIter):
                    total_infected, nbr_infected_at_iter = independent_cascade(G, generate_Set(G, j, i, model), i)
                    listTotInfected.append(total_infected)
                    for x in range(len(nbr_infected_at_iter)):
                        listTotInfected[x] += nbr_infected_at_iter[x]

                longest = 0
                for x in range(1, len(listTotInfected)):
                    if (listTotInfected[x] / nbrIter) < listTotInfected[x - 1]:
                        listTotInfected[x] = listTotInfected[x - 1]
                    else:
                        listTotInfected[x] = (listTotInfected[x] / nbrIter)
                        longest = max(longest, x)
                print(listModel[model], listTotInfected[:longest+1])
                plt.plot(range(0, len(listTotInfected)), listTotInfected, '-', label=  listModel[model])
                plt.xlim([0, longest+1])
                plt.xlabel("Number of iterations")
                plt.ylabel("Number of persons infected")
                plt.title("Evolution of the total number of infected persons with respect to the time (p =" + str(i) + ")")
            plt.legend()
            plt.show()


if "__main__":
    G = buildGraph()
    kCores = k_cores_decomposition(G)
    draw_cores(G, kCores)
    louvain_algorithm(G)
    for a, b in kCores.items():
        print(a, "core :", b)
    average_degree = []
    for node in G.nodes():
        average_degree.append(G.degree(node))
    print("Graphe normal : ")
    print("Moyenne des degrés :", np.mean(average_degree))
    print("Variance des degrés :", np.var(average_degree))
    print()
    GBarabasi = barabasi_albert_generation(G)
    average_degree = []
    for node in GBarabasi.nodes():
        average_degree.append(GBarabasi.degree(node))
    print("Graphe Barabasi : ")
    print("Moyenne des degrés :", np.mean(average_degree))
    print("Variance des degrés :", np.var(average_degree))
    print()
    k = [0.05]
    p = [0.05, 0.1, 0.2, 0.4]
    plot_Graph(G, k, p)
    #total_infected = []
    #new_infected = []
    #count = independent_cascade(G)
    #print("Number of iterations = ", count)
    #S = influence_maximization_problem(G)
    #S = inluence_maximization_problem_greedy(G)
    #print("Final set : ", S)
