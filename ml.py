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


trainFile = open('train.csv')
csvreaderTrain = csv.reader(trainFile)
train_dataset=pd.read_csv(trainFile)

testFile = open('test.csv')
csvreaderTest = csv.reader(testFile)
test_dataset=pd.read_csv(testFile)

train_dataset = train_dataset.replace(to_replace="Ghoul", value=1)
train_dataset = train_dataset.replace(to_replace="Goblin", value=2)
train_dataset = train_dataset.replace(to_replace="Ghost", value=3)

# y = all the Type names
y = train_dataset.iloc[:, 6].values

#TRAIN DATASET
train_dataset = train_dataset.drop(labels="id",axis=1)
train_dataset = train_dataset.drop(labels="type",axis=1)
train_dataset = train_dataset.replace(to_replace="clear", value=1/6)
train_dataset = train_dataset.replace(to_replace="green", value=2/6)
train_dataset = train_dataset.replace(to_replace="black", value=3/6)
train_dataset = train_dataset.replace(to_replace="blue", value=4/6)
train_dataset = train_dataset.replace(to_replace="white", value=5/6)
train_dataset = train_dataset.replace(to_replace="blood", value=1)

#TEST DATASET
test_dataset = test_dataset.drop(labels="id",axis=1)
test_dataset = test_dataset.replace(to_replace="clear", value=1/6)
test_dataset = test_dataset.replace(to_replace="green", value=2/6)
test_dataset = test_dataset.replace(to_replace="black", value=3/6)
test_dataset = test_dataset.replace(to_replace="blue", value=4/6)
test_dataset = test_dataset.replace(to_replace="white", value=5/6)
test_dataset = test_dataset.replace(to_replace="blood", value=1)


X_train, X_test, y_train, y_test = train_test_split(train_dataset, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = neighbors.KNeighborsClassifier(1)
classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
