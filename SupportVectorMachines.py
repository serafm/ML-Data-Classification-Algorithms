from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC




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
x_train,x_test,y_train,y_test = train_test_split(train_data,tlabel,test_size = 0.4, random_state=0)

# Linear SVM
svm_linear = LinearSVC()

svm_linear.fit(x_train,y_train)
train_pred_linear = svm_linear.predict(x_test)

# Accuracy and f1 score for Train data
acc_linear = accuracy_score(train_pred_linear, y_test)
score_linear = f1_score(train_pred_linear, y_test, average='weighted')

print("\n")
print("TRAIN LINEAR")
print("Accuracy= ", acc_linear)
print("f1 score(weighted)= ", score_linear)

#Predicting y for x_test
test_pred_linear = svm_linear.predict(test_data)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_pred_linear
new_csv.to_csv("TestReportSVMLinear.csv", index=False)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
output = pd.read_csv('TestReportSVMLinear.csv')
output = output.replace(to_replace=0, value="Ghoul")
output = output.replace(to_replace=1, value="Goblin")
output = output.replace(to_replace=2, value="Ghost")

new_csv = output
new_csv.to_csv("TestReportSVMLinear.csv", index=False)





# Gaussian SVM

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        svm_gaussian = SVC(C=C, gamma=gamma)
        svm_gaussian.fit(x_train,y_train)
        classifiers.append(svm_gaussian)

train_pred_gaussian = classifiers[8].predict(x_test)

#svm_gaussian = SVC(kernel='rbf', decision_function_shape = "ovr", C=10, gamma=0.5)

#svm_gaussian.fit(x_train,y_train)
#train_pred_gaussian = svm_gaussian.predict(x_test)

# Accuracy and f1 score for Train data
acc_gaussian = accuracy_score(train_pred_gaussian, y_test)
score_gaussian = f1_score(train_pred_gaussian, y_test, average='weighted')

print("\n")
print("TRAIN GAUSSIAN")
print("Accuracy= ", acc_gaussian)
print("f1 score(weighted)= ", score_gaussian)

#Predicting y for x_test
test_pred_gaussian = classifiers[8].predict(test_data)

# Make a csv file with id and type labels for Test prediction data
new_csv = pd.DataFrame()
new_csv["id"] = test_index
new_csv["type"] = test_pred_gaussian
new_csv.to_csv("TestReportSVMGaussian.csv", index=False)

# Replace type label from Sample data with numbers 0,1,2 (NOT NECESSARY)
output = pd.read_csv('TestReportSVMGaussian.csv')
output = output.replace(to_replace=0, value="Ghoul")
output = output.replace(to_replace=1, value="Goblin")
output = output.replace(to_replace=2, value="Ghost")

new_csv = output
new_csv.to_csv("TestReportSVMGaussian.csv", index=False)
