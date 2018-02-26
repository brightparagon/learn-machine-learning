import pandas
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# read csv file using pandas
csv = pandas.read_csv('iris.csv')

csv_data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
csv_label = csv["Name"]

# separate data
train_data, test_data, train_label, test_label = \
  train_test_split(csv_data, csv_label)

# make it learn and predict
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# result: percentage
ac_score = metrics.accuracy_score(test_label, pre)
print("right rate: ", ac_score)