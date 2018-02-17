import matplotlib.pyplot as plt
import pandas as pd
import json

# read data
with open('./lang/freq.json', "r", encoding="utf-8") as fp:
  freq = json.load(fp)

# calculate per language
lang_dic = {}
for i, lbl in enumerate(freq[0]["labels"]):
  fq = freq[0]["freqs"][i]
  if not (lbl in lang_dic):
    lang_dic[lbl] = fq
    continue
  for idx, v in enumerate(fq):
    lang_dic[lbl][idx] = (lang_dic[lbl][idx] + v) / 2

# put data in DataFrame of pandas
asclist = [[chr(n) for n in range(97, 97+26)]]
df = pd.DataFrame(lang_dic, index=asclist)

# paint a graph
plt.style.use('ggplot')
# df.plot(kind="bar", subplots=True, ylim=(0, 0.15))
df.plot(kind="line")
# plt.savefig("lang-plot.png")
plt.show()
