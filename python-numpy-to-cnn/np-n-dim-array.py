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

# broadcast in n-dim numpy array
B = np.array([10, 20])
print(A * B)

# access to numpy array
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0]) # first row of X
print(X[0][1])

# numpy array also can be accessed via loop
for row in X:
  print(row)

# flatten numpy array
X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])

# logical operation broadcast
print(X > 15)
print(X[X > 15])
