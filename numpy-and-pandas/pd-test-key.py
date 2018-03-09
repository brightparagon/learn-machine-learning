import pandas as pd

# height, weight and type
tbl = pd.DataFrame({
  "weight": [80.0, 70.4, 65.5, 45.9, 51.2],
  "height": [170, 180, 155, 143, 154],
  "type": ["f", "n", "n", "t", "t"]
})

# extract weight data
print("weight list")
print(tbl["weight"])

# extract weight and height
print("weight and height list")
print(tbl[["weight", "height"]])
