import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import csv
n_neighbors = 40

# import some data to play with
#iris = datasets.load_iris()
file = open('train.csv')

type(file)
csvreader = csv.reader(file)
data=list(csvreader)
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = data[:5]
data.pop(0)
#print(data)


h = 0.02  # step size in the mesh
color_matrix_names=["clear","green","black","blue","white","blood"]
color_matrix_values = [0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(6):
    color_matrix_values[i]=(i+1)/6

print(color_matrix_values)