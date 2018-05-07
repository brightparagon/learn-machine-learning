# tensorflow gitbook tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# it makes only one layer
# x is data: None rows(depends on the num of mnist data) with 784(28 * 28) columns
x = tf.placeholder(tf.float32, [None, 784])

# W is a parameter place: weight
W = tf.Variable(tf.zeros([784, 10]))
# b is a bias
b = tf.Variable(tf.zeros([10]))

# y is an answer from our layer
y = tf.nn.softmax(tf.matmul(x, W) + b)
# Y_ is a right answer label
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy holds information of how bad the result is
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# uses Gradient Descent strategy with 0.5 learning rate to minimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize variables of tensorflow
init = tf.global_variables_initializer()

# make a ssesion
sess = tf.Session()
sess.run(init)

for i in range(1000):
  # take 100 data randomly
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # then train with this mini batch of 100 data
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# array of true or false
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# array of true or false is casted to array of integer that then can be reduced to make a mean value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
