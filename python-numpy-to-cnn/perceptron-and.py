import numpy as np

def AND(x1, x2):
  # in this AND case actually theta would do its role when it's between 0.5, 0.999...
  # w1, w2, theta = 0.5, 0.5, 0.7
  # tmp = x1*w1 + x2*w2
  
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b

  if tmp <= 0:
    return 0
  else:
    return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
