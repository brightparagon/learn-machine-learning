import numpy as np

def softmax(x):
  constant = np.max(x)
  exp_a = np.exp(x - constant)
  sum_exp_a = np.sum(exp_a)

  y = exp_a / sum_exp_a

  return y
