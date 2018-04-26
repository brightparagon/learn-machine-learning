from dataset.mnist import load_mnist
import numpy as np
import pickle
from softmax import softmax
from sigmoid import sigmoid

def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test

def init_network():
  with open("./dataset/sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

  return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  # if x is one image
  # (1, 784) * (784, 50) -> (1, 50) * (50, 100) -> (1, 100) * (100, 10) -> (1, 10)
  # if x is 100 images
  # (100, 784) * (784, 50) -> (100, 50) * (50, 100) -> (100, 100) * (100, 10) -> (100, 10)
  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
  x_batch = x[i: i+batch_size] # 100, 784
  y_batch = predict(network, x_batch) # 100, 10
  p = np.argmax(y_batch, axis=1) # 100
  if i == 0:
    print(x_batch.shape, y_batch.shape, p.shape)
    print(p == t[i:i+batch_size])
    print(np.sum(p == t[i:i+batch_size]))
  accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
