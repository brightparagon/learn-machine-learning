import pandas as pd

# height, weight and type
tbl = pd.DataFrame({
  "weight": [80.0, 70.4, 65.5, 45.9, 51.2, 72.5],
  "height": [170, 180, 155, 143, 154, 160],
  "gender": ["f", "m", "m", "f", "f", "m"]
})

# normalize height and weight data
# get the max and min from data
def norm(tbl, key):
    c = tbl[key]
    v_max = c.max()
    v_min = c.min()
    print(key, "=", v_min, "-", v_max)
    tbl[key] = (c - v_min) / (v_max - v_min)

norm(tbl, "weight")
norm(tbl, "height")
print(tbl)
