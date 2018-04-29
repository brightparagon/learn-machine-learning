import numpy as np
from numerical_diff import *
from functions import *
from collections import OrderedDict
from affine_layer import Affine
from relu_layer import Relu
from softmax_with_loss_layer import SoftmaxWithLoss

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # initialize weight(dictionary)
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # create layers
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    # previous codes
    # W1, W2 = self.params['W1'], self.params['W2']
    # b1, b2 = self.params['b1'], self.params['b2']

    # a1 = np.dot(x, W1) + b1
    # z1 = sigmoid(a1)
    # a2 = np.dot(z1, W2) + b2
    # y = softmax(a2)

    # return y
    for layer in self.layers.values():
      x = layer.forward(x)
    
    return x

  # x: input data, t: answer label
  def loss(self, x, t):
    y = self.predict(x)

    # previous code
    # return cross_entropy_error(y, t)
    return self.lastLayer.forward(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    # t = np.argmax(t, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])

    return grads

  def gradient(self, x, t):
    # forward propagation
    self.loss(x, t)

    # backward propagation
    dout = 1
    dout self.lastLayer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # save results
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return gradss

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

# caculating these gradient by the numerical way takes too long time
grads = net.numerical_gradient(x, t)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
