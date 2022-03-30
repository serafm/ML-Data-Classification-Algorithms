import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import csv
import pandas as pd
n_neighbors = 40

# import some data to play with
#iris = datasets.load_iris()
file = open('train.csv')
field_names=['bone-length','rotting_flesh','hair_length','has-soul','color']
type(file)
csvreader = csv.reader(file)
dataset=pd.read_csv(file)
print(dataset.head())


"""""
h = 0.02  # step size in the mesh
color_matrix_names=["clear","green","black","blue","white","blood"]
color_matrix_values = [0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(6):
    color_matrix_values[i]=(i+1)/6
clf=neighbors.KNeighborsClassifier(n_neighbors,"distance")
print(color_matrix_values)"""""