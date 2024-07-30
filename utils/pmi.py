from sklearn import decomposition
from numpy.linalg import inv
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import scipy.sparse as sp
import sklearn.preprocessing as nor
from copy import deepcopy
from munkres import Munkres
from sklearn.cluster import KMeans

from utils.NetMF import NetMF

# ==========================计算PMI矩阵=======================
# from karateclub import NetMF
model = NetMF()


def getPMI_Matrix(W):
    W = nx.Graph(W)
    W = model._create_target_matrix(W)
    W = W.toarray()
    return W


# =======================计算度矩阵===================
def calDegreeMatrix(adjacentMatrix):
    degreeMatrix_list = np.sum(adjacentMatrix, axis=1)  # 得到一个列表，列表中的值是每一行的和
    DegreeMatrix = np.diag(degreeMatrix_list)
    return DegreeMatrix


# ==================计算拉普拉斯矩阵================
def calLaplacianMatrix(adjacentMatrix):
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)  # 得到一个列表，列表中的值是每一行的和
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
    return laplacianMatrix


# ================对行进行L1正则化==================
def cal_sim(s):
    re = nor.normalize(s, axis=1, norm="l1")
    return re
