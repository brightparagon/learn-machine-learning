from softmax import softmax
from cee import cross_entropy_error

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None # output of softmax
    self.t = None # awnser label

  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx
