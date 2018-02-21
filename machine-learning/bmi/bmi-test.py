from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# read weight and height data
tbl = pd.read_csv("bmi.csv")

# normalize data by colums
label = tbl["label"]
w = tbl["weight"] / 100 # presume max weight is 100kg
h = tbl["height"] / 200 # presume max height is 200cm
wh = pd.concat([w, h], axis=1)

# split data into two parts: learn & test
data_train, data_test, label_train, label_test = \
  train_test_split(wh, label)

# make it learn
clf = svm.SVC()
clf.fit(data_train, label_train)

# predict
predict = clf.predict(data_test)

# result
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("right rate: ", ac_score)
print("report :\n", cl_report)
