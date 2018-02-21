import random

# function returning a label based on weight and height
def calc_bmi(h, w):
  bmi = w / (h/100) ** 2
  if bmi < 18.5: return "thin"
  if bmi < 25: return "normal"
  return "fat"

# write a file
fp = open("bmi.csv", "w", encoding="utf-8")
fp.write("height,weight,label\r\n")

# create data randomly
cnt = {"thin": 0, "normal": 0, "fat": 0}
for i in range(20000):
  h = random.randint(120, 200)
  w = random.randint(35, 80)
  label = calc_bmi(h, w)
  cnt[label] += 1
  fp.write("{0},{1},{2}\r\n".format(h, w, label))
fp.close()

print("done, ", cnt)
