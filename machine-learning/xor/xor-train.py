from sklearn import svm

xor_data = [
  #P, Q, result
  [0, 0, 0],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 0]
]

# separate xor_data into data and label
data = []
label = []
for row in xor_data:
  p = row[0]
  q = row[1]
  r = row[2]
  data.append([p, q])
  label.append(r)

print("data: ", data)

# make it learn using SVM
clf = svm.SVC()
clf.fit(data, label)

# predict
pre = clf.predict(data)
print("prediction: ", pre)

# result: percentage
ok = 0; total = 0
for index, answer in enumerate(label):
  p = pre[index]
  if p == answer: ok += 1
  total += 1
print("right answer percentage: ", ok, "/", total, "=", ok / total)
