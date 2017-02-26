# 2017-data-science-training2-titanic

import pandas as pd
import numpy as np
import csv as csv
import copy
from sklearn import tree

train = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv', header =0)
test = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv', header =0)

# Convert sex to integer form, female =1, male =0
train["Gender"] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# handle missing age with median
train["Age"] = train["Age"].fillna(train.Age.median())

print("passenger no.6 who has missing age")
print(train.Name[5]) # to check if missing age is filled
print(train.Age[5]) # to check if missing age is filled

# handle missing Embarked with S 
#Convert Embarked to integer, S =0, C = 1, Q =2
train["Embarked"] = train["Embarked"].fillna('S')
train["Embarked2"] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train.Embarked2[5]) # to check if Embarked is correctly converted
target = train["Survived"].values
features_one = train[["Gender", "Age", "Embarked2"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print("")
print("feature importances are:")
print(my_tree_one.feature_importances_)
print("overall feature score is:")
print(my_tree_one.score(features_one, target))

#need to convert the features for test file 
test["Gender"] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test["Age"] = test["Age"].fillna(train.Age.median())
test["Embarked"] = test["Embarked"].fillna('S')
test["Embarked2"] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Gender", "Age", "Embarked2"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print("")
print("my prediction:")
print(my_solution)

prediction_file = open("./Ali_titanic.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId","Survived"])
