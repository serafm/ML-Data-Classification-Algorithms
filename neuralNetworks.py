import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Read the csv files
train_data = pd.read_csv('data/train.csv')
test_data  = pd.read_csv('data/test.csv')

# Replace type label from Train data with numbers 0,1,2 (NOT NECESSARY)
train_data = train_data.replace(to_replace="Ghoul", value=0)
train_data = train_data.replace(to_replace="Goblin", value=1)
train_data = train_data.replace(to_replace="Ghost", value=2)

# Get label name: Type from Sample data and Train data 
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
classifier = MLPClassifier(hidden_layer_sizes=(k), max_iter=300, activation = 'logistic', solver='sgd', alpha=0.0001, random_state=1)

#Fitting the training data to the network
classifier.fit(x_train, y_train)

#Predicting y for x_test
train_pred = classifier.predict(x_test)

# Accuracy and f1 score for Train data
acc = accuracy_score(train_pred, y_test)
score = f1_score(train_pred, y_test, average='weighted')

print("\n")
print("TRAIN")
print("Accuracy= ", acc)
print("f1 score(weighted)= ", score)

#Predicting y for x_test
test_pred = classifier.predict(test_data)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_pred
new_csv.to_csv("TestReportNeuralNetwork.csv", index=False)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
output = pd.read_csv('TestReportNeuralNetwork.csv')
output = output.replace(to_replace=0, value="Ghoul")
output = output.replace(to_replace=1, value="Goblin")
output = output.replace(to_replace=2, value="Ghost")

new_csv = output
new_csv.to_csv("TestReportNeuralNetwork.csv", index=False)


#B WAY

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

print("\n")
print("TRAIN for B")
print("Accuracy= ", accB)
print("f1 score(weighted)= ", scoreB)
print("\n")

#Predicting y for x_test
test_predB = classifierB.predict(test_data)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_predB
new_csv.to_csv("TestReportNeuralNetworkB.csv", index=False)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
output = pd.read_csv('TestReportNeuralNetworkB.csv')
output = output.replace(to_replace=0, value="Ghoul")
output = output.replace(to_replace=1, value="Goblin")
output = output.replace(to_replace=2, value="Ghost")

new_csv = output
new_csv.to_csv("TestReportNeuralNetworkB.csv", index=False)
