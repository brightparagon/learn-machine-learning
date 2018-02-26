import pandas as pd
from sklearn import model_selection, svm, metrics
from sklearn.grid_search import GridSearchCV

# read mnist data
train_csv = pd.read_csv("./mnist/train.csv")
test_csv = pd.read_csv("./mnist/t10k.csv")

# extrach cols needed
train_label = train_csv.ix[:, 0]
train_data = train_csv.ix[:, 1:577]
test_label = test_csv.ix[:, 0]
test_data = test_csv.ix[:, 1:577]
print("the number of train data: ", len(train_label))

# set parameters for grid search
params = [
  {"C": [1,10,100,1000], "kernel": ["linear"]},
  {"C": [1,10,100,1000], "kernel": ["rbf"], "gamma": [0.001,0.0001]}
]

# do grid search
clf = GridSearchCV( svm.SVC(), params, n_jobs=-1 )
clf.fit(train_data, train_label)
print("trainer: ", clf.best_estimator_)

# check test data
pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(pre, test_label)
print("right rate: ", ac_score)
