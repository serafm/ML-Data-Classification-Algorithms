import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import csv
import pandas as pd
n_neighbors = 40

# import some data to play with
#iris = train_datasets.load_iris()
trainFile = open('train.csv')
csvreaderTrain = csv.reader(trainFile)
train_dataset=pd.read_csv(trainFile)

testFile = open('test.csv')
csvreaderTest = csv.reader(testFile)
test_dataset=pd.read_csv(testFile)

#print("TRAIN DATASET")
train_dataset = train_dataset.drop(labels="id",axis=1)
train_dataset = train_dataset.drop(labels="type",axis=1)
train_dataset = train_dataset.replace(to_replace="clear", value=1/6)
train_dataset = train_dataset.replace(to_replace="green", value=2/6)
train_dataset = train_dataset.replace(to_replace="black", value=3/6)
train_dataset = train_dataset.replace(to_replace="blue", value=4/6)
train_dataset = train_dataset.replace(to_replace="white", value=5/6)
train_dataset = train_dataset.replace(to_replace="blood", value=1)

y = train_dataset.iloc[:, 4].values
y=y.astype('int')


#print(train_dataset.head())

#print("\n")

#print("TEST DATASET")
test_dataset = test_dataset.drop(labels="id",axis=1)
test_dataset = test_dataset.replace(to_replace="clear", value=1/6)
test_dataset = test_dataset.replace(to_replace="green", value=2/6)
test_dataset = test_dataset.replace(to_replace="black", value=3/6)
test_dataset = test_dataset.replace(to_replace="blue", value=4/6)
test_dataset = test_dataset.replace(to_replace="white", value=5/6)
test_dataset = test_dataset.replace(to_replace="blood", value=1)

y_test = test_dataset.iloc[:, 4].values
y_test=y_test.astype('int')

#print(test_dataset.head())

scaler = StandardScaler()
scaler.fit(train_dataset)
train_dataset = scaler.transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

classifier = neighbors.KNeighborsClassifier(1)
classifier.fit(train_dataset,y)

y_predict = classifier.predict(test_dataset)
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))

#X=train_dataset.iloc[0:,:-1]
#print(X)




"""""
h = 0.02  # step size in the mesh
color_matrix_names=["clear","green","black","blue","white","blood"]
color_matrix_values = [0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(6):
    color_matrix_values[i]=(i+1)/6
clf=neighbors.KNeighborsClassifier(n_neighbors,"distance")
print(color_matrix_values)"""""