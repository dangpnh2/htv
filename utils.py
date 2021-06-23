import numpy as np  
from sklearn.neighbors import KNeighborsClassifier
import csv
import pickle
import collections
import os
import pandas as pd

def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))

    return output

def create_dirs():
    import os

    names = ['bbc', 'reuters', '20news', 'mendeley']
    list_idx = [str(i) for i in range(10)]

    for name in names:
        path = 'experiments/'+name+'/inv/'
        for i in list_idx:
            os.makedirs(path+i+"/")