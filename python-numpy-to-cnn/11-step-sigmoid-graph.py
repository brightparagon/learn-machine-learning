import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y_step = step_function(x)
y_sig = sigmoid(x)

# print(y)
plt.plot(x, y_sig)
plt.show()
