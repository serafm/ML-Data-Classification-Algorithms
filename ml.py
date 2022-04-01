import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import csv
import pandas as pd



# Train dataset
trainFile = open('data/train.csv')
train_dataset=pd.read_csv(trainFile)

train_dataset = train_dataset.replace(to_replace="Ghoul", value=1)
train_dataset = train_dataset.replace(to_replace="Goblin", value=2)
train_dataset = train_dataset.replace(to_replace="Ghost", value=3)

# Test dataset 
testFile = open('data/test.csv')
test_dataset=pd.read_csv(testFile)

# Sample dataset 
sampleFile = open('data/sample_submission.csv')
sample_dataset=pd.read_csv(sampleFile)
sample_dataset = sample_dataset.replace(to_replace="Ghost", value=3)

# y = all the Type names
y = train_dataset.iloc[:, 6].values

# sample = all the Type names
sample = sample_dataset.iloc[:, 1].values


#TRAIN DATASET
train_dataset = train_dataset.drop(labels="id",axis=1)
train_dataset = train_dataset.drop(labels="type",axis=1)
train_dataset = train_dataset.replace(to_replace="clear", value=1 * 0.167)
train_dataset = train_dataset.replace(to_replace="green", value=2 * 0.167)
train_dataset = train_dataset.replace(to_replace="black", value=3* 0.167)
train_dataset = train_dataset.replace(to_replace="blue", value=4 *  0.167)
train_dataset = train_dataset.replace(to_replace="white", value=5 * 0.167)
train_dataset = train_dataset.replace(to_replace="blood", value=1)
print("\n")

#TEST DATASET
test_dataset = test_dataset.drop(labels="id",axis=1)
test_dataset = test_dataset.replace(to_replace="clear", value=1*0.167)
test_dataset = test_dataset.replace(to_replace="green", value=2*0.167)
test_dataset = test_dataset.replace(to_replace="black", value=3*0.167)
test_dataset = test_dataset.replace(to_replace="blue", value=4*0.167)
test_dataset = test_dataset.replace(to_replace="white", value=5*0.167)
test_dataset = test_dataset.replace(to_replace="blood", value=1)

X_train, X_test, y_train, y_test = train_test_split(train_dataset, y, test_size=0.2)


# TRAIN STATISTICS
print("\n                         " + '\x1b[6;30;42m' + "TRAIN STATS"  + '\x1b[0m' + "\n")
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = neighbors.KNeighborsClassifier(5)
classifier.fit(X_train,y_train)

train_predict = classifier.predict(X_test)



# TEST STATISTICS
print("\n                         " + '\x1b[6;30;42m' + "TEST STATS"  + '\x1b[0m' + "\n")


test_predict = classifier.predict(test_dataset)
print(classifier.score(sample,test_predict))

"""
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = final_pred
new_csv.to_csv("submission.csv",index=False)"""


