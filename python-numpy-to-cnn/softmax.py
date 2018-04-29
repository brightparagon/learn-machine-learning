import numpy as np

# def softmax(x):
#   constant = np.max(x)
#   exp_a = np.exp(x - constant)
#   sum_exp_a = np.sum(exp_a)

#   y = exp_a / sum_exp_a

#   return y

def softmax(x):
  if x.ndim == 2:
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 

  x = x - np.max(x) # avoid overflow
  return np.exp(x) / np.sum(np.exp(x))
