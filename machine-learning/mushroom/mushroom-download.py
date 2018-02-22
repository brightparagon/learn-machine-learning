import urllib.request as req
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
local = "mushroom.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
req.urlretrieve(url, local)
print("done")
