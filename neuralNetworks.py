import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


sc = StandardScaler()

# Read the csv files
train_data = pd.read_csv('data/train.csv')
test_data  = pd.read_csv('data/test.csv')
sample_data = pd.read_csv('data/sample_submission.csv')

# Replace type label from Train data with numbers 0,1,2 (NOT NECESSARY)
train_data = train_data.replace(to_replace="Ghoul", value=0)
train_data = train_data.replace(to_replace="Goblin", value=1)
train_data = train_data.replace(to_replace="Ghost", value=2)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
sample_data = sample_data.replace(to_replace="Ghoul", value=0)
sample_data = sample_data.replace(to_replace="Goblin", value=1)
sample_data = sample_data.replace(to_replace="Ghost", value=2)

# Get label name: Type from Sample data and Train data 
expected_output = sample_data['type']
tlabel = train_data['type']

# remove lables: type,id from Train data  and replace colors with numbers between [0,1]
train_data.describe()
train_data = train_data.drop('type',1)
train_data = train_data.drop('id',1)
train_data = train_data.replace(to_replace="clear", value=1/6)
train_data = train_data.replace(to_replace="green", value=2/6)
train_data = train_data.replace(to_replace="black", value=3/6)
train_data = train_data.replace(to_replace="blue", value=4/6)
train_data = train_data.replace(to_replace="white", value=5/6)
train_data = train_data.replace(to_replace="blood", value=1)

# Set test_index = id (we are going to use it later to create the csv file)
test_index = test_data['id']

# Remove lable: id from Test data
test_data.drop('id',1,inplace=True)

# Replace Test data colors with numbers between [0,1]
test_data = test_data.replace(to_replace="clear", value=1/6)
test_data = test_data.replace(to_replace="green", value=2/6)
test_data = test_data.replace(to_replace="black", value=3/6)
test_data = test_data.replace(to_replace="blue", value=4/6)
test_data = test_data.replace(to_replace="white", value=5/6)
test_data = test_data.replace(to_replace="blood", value=1)

# Set data for train and test from the Train data 
x_train,x_test,y_train,y_test = train_test_split(train_data,tlabel,test_size = 0.2,random_state=1)


#Initializing the MLPClassifier
k = 100
classifier = MLPClassifier(hidden_layer_sizes=(k), max_iter=300, activation = 'logistic', solver='sgd', random_state=1)

#Fitting the training data to the network
classifier.fit(x_train, y_train)

#Predicting y for x_test
y_pred = classifier.predict(x_test)


# Accuracy and f1 score for Train data
acc = accuracy_score(y_pred, y_test)
score = f1_score(y_pred, y_test, average='weighted')

print("TRAIN")
print("Accuracy= ", acc)
print("f1 score(weighted)= ", score)


#Predicting y for x_test
test_pred = classifier.predict(test_data)

# Accuracy and f1 score for Train data
acc2 = accuracy_score(test_pred, expected_output)
score2 = f1_score(test_pred, expected_output, average='weighted')

print("TEST")
print("Accuracy= ", acc2)
print("f1 score(weighted)= ", score2)


#B

#Initializing the MLPClassifier
k1 = 200
k2 = 100
classifierB = MLPClassifier(hidden_layer_sizes=(k1,k2), max_iter=300, activation = 'logistic', solver='sgd', random_state=1)

#Fitting the training data to the network
classifierB.fit(x_train, y_train)

#Predicting y for x_test
y_predB = classifierB.predict(x_test)

# Accuracy and f1 score for Train data
accB = accuracy_score(y_predB, y_test)
scoreB = f1_score(y_predB, y_test, average='weighted')

print("TRAIN for B")
print("Accuracy= ", accB)
print("f1 score(weighted)= ", scoreB)

#Predicting y for x_test
test_predB = classifierB.predict(test_data)

# Accuracy and f1 score for Train data
acc2B = accuracy_score(test_predB, expected_output)
score2B = f1_score(test_predB, expected_output, average='weighted')

print("TEST for B")
print("Accuracy= ", acc2B)
print("f1 score(weighted)= ", score2B)





     