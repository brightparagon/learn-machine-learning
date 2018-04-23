import numpy as np

# sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# identity function
def identity(x):
  return x

# first layer
X = np.array([1.0, 0.5]) # 1, 2
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2, 3
B1 = np.array([0.1, 0.2, 0.3]) # 1, 3

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1 # 1, 3
Z1 = sigmoid(A1) # 1, 3
print(A1)
print(Z1)

# second layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3, 2
B2 = np.array([0.1, 0.2]) # 1, 2

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2 # 1, 2
Z2 = sigmoid(A2) # 1, 2

# last layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2, 2
B3 = np.array([0.1, 0.2]) # 1, 2

A3 = np.dot(Z2, W3) + B3 # 1, 2
Y = identity(A3) # 1, 2

print(Y)
