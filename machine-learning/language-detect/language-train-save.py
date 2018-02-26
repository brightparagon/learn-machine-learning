from sklearn import svm
from sklearn.externals import joblib
import json

# read frequencies of words for each language
with open("./lang/freq.json", "r", encoding="utf-8") as fp:
  d = json.load(fp)
  data = d[0]

# make it learn
clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])

# save the result of learning
joblib.dump(clf, "./lang/freq.pkl")
print("done")
