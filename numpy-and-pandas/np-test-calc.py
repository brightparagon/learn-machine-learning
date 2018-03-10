import numpy as np

# create ten float data
v = np.zeros(10, dtype=np.float32)
print(v)

# create ten continuous uint64 data
v = np.arange(10, dtype=np.uint64)
print(v)

# multiply v by 3
v *= 3
print(v)

# get mean of v
print(v.mean())
