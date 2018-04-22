import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])

# element-wise arithmetic operation in numpy array
print(A + B)
print(A * B)

# matrix * scala: broadcast
print(A)
print(A * 10)
