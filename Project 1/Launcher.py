from pprint import pprint
import matplotlib.pyplot as plt
from collections import Counter

import networkx as nx
import community
import pandas as pd
import sys

listOfName = ["Pavlovna", "Vasili Kuragin", "Helene", "Pierre", "Hippolyte", "Zherkov", "Captain Timokhin", "Alpatych", "Weyrother",
              "Mortemart", "Morio", "Bolkonskaya", "Nicholas", "Joseph Alexeevich", "Peronskaya", "Vasili Dmitrich", "Tikhon", "Dolgorukov", "Langeron",
              "Mikhaylovna", "Anatole Kuragin",  "Dolokhov", "Stevens" ,  "Countess Rostova", "Denisov", "Likhachev","Lavrushka", "Miloradovich", "The Emperor",
              "Count Ilya Rostov", "Count Rostov",  "Vera Rostova", "Nikolai Rostov",  "Natasha", "Petya",  "Sonya", "Drubetskoy", "Bonaparte", "Kirsten",
              "Dmitri", "Marya Lvovna Karagina", "Karagina", "Count Cyril", "Princess Katerina Mamontova", "Bilibin","Captain Tushin", "Repnin","Bourienne",
              "Shinshin", "Julie Drubetskaya", "Jacquot", "Márya Dmítrievna", "Kuzmich", "Marya Fedorovna", "Anisya Fedorovna"] #Liste non exhaustive des personnages, 52 personnages identifiés

'''
Pre : /
Post : Crée le graphe reliant les personnages entre eux si ils sont présents dans le même paragraphe
'''
def buildGraph():
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
                line = castLine(line)#retire les caracteres spéciaux
                for name in listOfName:
                    if name == "Count Ilya Rostov":#La moitie des apparition sont sous le nom Count Ilya Rostov, l'autre moitie Count Rostov
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

'''
Pre : /
Post: Les caracteres accentues de line ont ete retire
'''
def castLine(line):
    line = line.replace("é","e")
    line = line.replace("ë","e")
    line = line.replace("è","e")
    line = line.replace("à","a")
    line = line.replace("á","a")
    line = line.replace("ó","o")
    line = line.replace("í","i")
    line = line.replace("ú","u")
    return line

'''
Pre : G est un graphe, listName une liste de nom apparaissant dans le meme paragraphe
Post : Une arrete a ete ajoutee dans G pour toutes les paires de nom dans listName
'''
def addToGraph(G, listName):
    for i in range(len(listName)):
        for j in range(i+1, len(listName)):
            G.add_edge(listName[i], listName[j])

def Q1(G):
    pass

'''
Pre : G est un graphe
Post : Calcule les k-cores de G, et les renvoies sous forme de dictionnaire sous la forme {k, nodes_in_k_core}
'''
def Q2(G):
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



def Q3(G):
    pass

if "__main__":
    G = buildGraph()
    nx.draw(G, node_size = 100)
    plt.show()
    Q1(G)
    kCores = Q2(G)
    for a,b in kCores.items():
        print(a,"core :", b)
    Q3(G)
