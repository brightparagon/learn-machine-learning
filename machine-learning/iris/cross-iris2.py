import pandas as pd
from sklearn import svm, metrics, model_selection
import random, re

# read data from iris csv file
csv = pd.read_csv('iris.csv')

# separate the data to two parts: train and test
data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

# cross validation
clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)

print("each right rate: ", scores)
print("average right rate: ", scores.mean())
