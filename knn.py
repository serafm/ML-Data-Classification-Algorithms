import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


# Read the csv files
train_data = pd.read_csv('train.csv')
test_data  = pd.read_csv('test.csv')
sample_data = pd.read_csv('sample_submission.csv')

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

# Set data for train and test from the Train data 
x_train,x_test,y_train,y_test = train_test_split(train_data,tlabel,test_size = 0.5,random_state=0)

# KNN Algorithm with k neighbors
k = 3
knn = neighbors.KNeighborsClassifier(k)
knn.fit(x_train,y_train)

# Prediction for Train data
prediction = knn.predict(x_test)

# Print the statistics from training
print("\n                         " + '\x1b[6;30;42m' + "Train Statistics"  + '\x1b[0m' + "\n")
print(classification_report(prediction,y_test))

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

# Prediction for Test data
test_prediction = knn.predict(test_data)

# Accuracy and f1 score for Test data
acc = accuracy_score(expected_output, test_prediction)
score = f1_score(expected_output, test_prediction, average='weighted')

# Print accuracy and f1 score of Test data
print("\n                  " + '\x1b[6;30;42m' + "Test Accuracy and f1 score"  + '\x1b[0m' + "\n")
print("Accuracy= ", acc)
print("f1 score= ", score)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_prediction
new_csv.to_csv("Test Report.csv", index=False)