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
  exdata = []
  for col, v in enumerate(row.ix[1:]):
    if row_index == 0:
      attr = {"dic": {}, "cnt": 0}
      attr_list.append(attr)
    else:
      attr = attr_list[col]
    # represent signs of mushrooms as arrays
    d = [0,0,0,0,0,0,0,0,0,0,0,0]
    if v in attr["dic"]:
      idx = attr["dic"][v]
    else:
      idx = attr["cnt"]
      attr["dic"][v] = idx
      attr["cnt"] += 1
    d[idx] = 1
    exdata += d
  data.append(exdata)

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
print("right rate: ", ac_score)
