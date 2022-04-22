import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# Read the csv files
train_data = pd.read_csv('data/train.csv')
test_data  = pd.read_csv('data/test.csv')

# Replace type label from Train data with numbers 0,1,2 (NOT NECESSARY)
train_data = train_data.replace(to_replace="Ghoul", value=0)
train_data = train_data.replace(to_replace="Goblin", value=1)
train_data = train_data.replace(to_replace="Ghost", value=2)

# Get label name: Type from Train data 
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
x_train,x_test,y_train,y_test = train_test_split(train_data,tlabel,test_size = 0.25,random_state=0)

# KNN Algorithm with Euclidean Distance
k = 10
knn = neighbors.KNeighborsClassifier(k, weights="distance")
knn.fit(x_train,y_train)

# Prediction for Train data
prediction = knn.predict(x_test)

# Print the statistics from training
print("\n                         " + '\x1b[6;30;42m' + "Train Statistics"  + '\x1b[0m' + "\n")
#print(classification_report(prediction,y_test))

acc = accuracy_score(prediction, y_test)
score = f1_score(prediction, y_test, average='weighted')
print("Accuracy=", acc)
print("F1 score=", score)

# Prediction for Test data
test_prediction = knn.predict(test_data)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_prediction
new_csv.to_csv("TestReportKNN.csv", index=False)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
output = pd.read_csv('TestReportKNN.csv')
output = output.replace(to_replace=0, value="Ghoul")
output = output.replace(to_replace=1, value="Goblin")
output = output.replace(to_replace=2, value="Ghost")

new_csv = output
new_csv.to_csv("TestReportKNN.csv", index=False)
