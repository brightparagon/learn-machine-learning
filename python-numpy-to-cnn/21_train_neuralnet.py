import numpy as np
from dataset.mnist import load_mnist
from twolayernet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# hyper parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100 # mini batch size
learning_rate = 0.1

# number of iteration per epoch
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num): # iterate 10,000 times
  # get mini batch
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # calculate gradients in a numerical way
  # grad = network.numerical_gradient(x_batch, t_batch)
  grad = network.gradient(x_batch, t_batch)

  # update weights
  for key in ('W1', 'b1', 'W2', 'b2'): # tuple is also iterable in python
    # network.params => weights
    network.params[key] -= learning_rate * grad[key]

  # save every loss: it should go down as training goes
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  # save accuracy info per epoch
  if i % iter_per_epoch == 0: # in this case iter_per_epoch is 600
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# print("test case: " + network.predict(x_test[0]))
# print("answer: " + t_test[0])
