from sklearn import svm, metrics
import random, re

# read data from iris.csv file
csv = []
with open('iris.csv', 'r', encoding='utf-8') as fp:
  # read a line
  for line in fp:
    line = line.strip() # get rid of \n
    cols = line.split(',') # differentiate data by comma
    # convert character data to number data
    fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
    cols = list(map(fn, cols))
    csv.append(cols)

# delete header
del csv[0]

# shuffle data
random.shuffle(csv)

# separate data into two parts: data for learning and testing
total_len = len(csv)
train_len = int(total_len * 2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
  data = csv[i][0:4] # colums from 0 to 3
  label = csv[i][4] # iris name(right answer)
  if i < train_len: # train data
    train_data.append(data)
    train_label.append(label)
  else: # test data
    test_data.append(data)
    test_label.append(label)
  
# make it learn and predict
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# result: percentage
ac_score = metrics.accuracy_score(test_label, pre)
# print("right answer percentage: ", ac_score)

for i in range(len(test_label)):
  print(test_label[i], ' : ', pre[i])
