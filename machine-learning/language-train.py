from sklearn import svm, metrics
import glob, os.path, re, json

# read texts and calculate the frequency of occurence of words
def check_freq(fname):
  name = os.path.basename(fname)
  lang = re.match(r'^[a-z]{2,}', name).group()
  with open(fname, "r", encoding="utf-8") as f:
    text = f.read()
  text = text.lower()

  cnt = [0 for n in range(0, 26)]
  code_a = ord("a")
  code_z = ord("z")

  # count each alphabet
  for ch in text:
    n = ord(ch)
    if code_a <= n <= code_z: # between a and z
      cnt[n - code_a] += 1
  
  # normalize
  total = sum(cnt)
  freq = list(map(lambda n: n / total, cnt))
  return (freq, lang)

def load_files(path):
  freqs = []
  labels = []
  file_list = glob.glob(path)
  for fname in file_list:
    r = check_freq(fname)
    freqs.append(r[0])
    labels.append(r[1])
  return {"freqs": freqs, "labels": labels}

data = load_files("./lang/train/*.txt")
test = load_files("./lang/test/*.txt")

# save it as json files
with open("./lang/freq.json", "w", encoding="utf-8") as fp:
  json.dump([data, test], fp)

# make it learn
clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])

# predict
predict = clf.predict(test["freqs"])

# result
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("right rate: ", ac_score)
print("report: ")
print(cl_report)
