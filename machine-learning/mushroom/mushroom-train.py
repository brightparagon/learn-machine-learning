import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# read data
mr = pd.read_csv("mushroom.csv", header=None)

# convert signs to numbers
label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
  label.append(row.ix[0])
  row_data = []
  for v in row.ix[1:]:
    row_data.append(ord(v))
  data.append(row_data)

# split data into two parts: train and test
data_train, data_test, label_train, label_test = \
  train_test_split(data, label)

# make it learn
clf = RandomForestClassifier()
clf.fit(data_train, label_train)

# predict
predict = clf.predict(data_test)

# result
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("right rate: ", ac_score)
print("report: ", cl_report)
