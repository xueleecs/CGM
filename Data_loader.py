import networkx as nx
import numpy as np
from Jaccard import *
def readSeq(ID):

    Path = "/fasta/"
    filename = Path + ID + ".fasta"
    fr = open(filename)
    next(fr)
    Seq = fr.read().replace("\n", "")
    fr.close()
    return Seq


def load(fileName):
    fileName = fileName
    data = []

    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        # print(curLine)
        lineArr = (curLine[0], curLine[1])
        data.append(lineArr)
    fr.close()
    return data

def loadtt(fileName):
    data = []

    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        lineArr = [curLine[0], curLine[1], int(curLine[2])]
        data.append(lineArr)
    return data


def loadGdata(fileName):
    data = []

    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        if curLine[2] == '1':
            lineArr = [curLine[0], curLine[1]]
            data.append(lineArr)
    return data

def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(element)
        else:
            d.extend(getnewList(element))
    return d

def Init22(Test_File, Train_File, proteinID):
    print("DataShape......")
    g_train = loadGdata(Train_File)
    g_test = loadGdata(Test_File)
    g_data = g_train + g_test
    g = nx.Graph()
    g.add_nodes_from(proteinID)
    g.add_edges_from(g_data)
    nodelist = list(g.nodes())
    Nodecount = len(nodelist)
    nodedic = dict(zip(list(g.nodes()), range(Nodecount)))  
    MatrixAdjacency_Train = np.zeros([Nodecount, Nodecount], dtype=np.int32)
    print("Nodecount  " + str(Nodecount))
    nodedic = dict(zip(list(g.nodes()), range(Nodecount)))
    for i in range(len(g_train)):
 
        MatrixAdjacency_Train[nodedic[g_train[i][0]], nodedic[g_train[i][1]]] = 1
        MatrixAdjacency_Train[nodedic[g_train[i][1]], nodedic[g_train[i][0]]] = 1

    Matrix_similarity = Jaccard(MatrixAdjacency_Train)

    where_are_nan = np.isnan(Matrix_similarity)
    Matrix_similarity[where_are_nan] = -1

    return Matrix_similarity, nodedic, nodelist

