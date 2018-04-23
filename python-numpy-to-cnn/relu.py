import numpy as np

# relu picks a bigger value between a given x and zero
def relu(x):
  return np.maximum(0, x)
