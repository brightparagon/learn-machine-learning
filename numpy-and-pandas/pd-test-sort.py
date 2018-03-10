import pandas as pd

# height, weight and type
tbl = pd.DataFrame({
  "weight": [80.0, 70.4, 65.5, 45.9, 51.2, 72.5],
  "height": [170, 180, 155, 143, 154, 160],
  "gender": ["f", "m", "m", "f", "f", "m"]
})

print("sort by height")
print(tbl.sort_values(by="height"))

print("sort by weight")
print(tbl.sort_values(by="weight"))
