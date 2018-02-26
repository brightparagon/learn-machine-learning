from sklearn import model_selection, svm, metrics

# read csv files and preprocess it
def load_csv(fname):
  labels = []
  images = []

  with open(fname, "r") as f:
    for line in f:
      cols = line.split(",")
      if len(cols) < 2: continue
      labels.append(int(cols.pop(0)))
      vals = list(map(lambda n: int(n) / 256, cols))
      images.append(vals)
  return {"labels": labels, "images": images}

data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

# make it learn
clf = svm.SVC()
clf.fit(data["images"], data["labels"])

# predict
predict = clf.predict(test["images"])

# result
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("right rate: ", ac_score)
print("report: ")
print(cl_report)
