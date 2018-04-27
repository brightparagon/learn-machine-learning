import numpy as np
from gradient_descent import gradient_descent

def function_2(x):
  return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))
# [-6.11110793e-10  8.14814391e-10]
# both are almost close to zero
# which we can say it arrives at its the smallest point of a graph of y = x(1)**2 + x(2)**2
