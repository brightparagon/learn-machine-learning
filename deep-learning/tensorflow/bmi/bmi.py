import tensorflow as tf
import numpy as np
import pandas as pd

# read data from csv file: height, weight and label
csv = pd.read_csv("bmi.csv")

# normalize the data above
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# convert label to array
# thin=(1,0,0) / normal=(0,1,0) / fat=(0,0,1)
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x]))

# searate data
test_csv = csv[15000:20000]
test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])

# define data flow
# define placeholder
x = tf.placeholder(tf.float32, [None, 2]) # height and weight
y_ = tf.placeholder(tf.float32, [None, 3]) # label

# define variables
W = tf.Variable(tf.zeros([2, 3])) # weight(not equal to the above)
b = tf.Variable(tf.zeros([3])) # bias

# define softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# train the model
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# get the right rate
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# start a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for step in range(3500):
  i = (step * 100) % 14000
  rows = csv[1 + i : 1 + i + 100]
  x_pat = rows[["weight","height"]]
  y_ans = list(rows["label_pat"])
  fd = {x: x_pat, y_: y_ans}
  sess.run(train, feed_dict=fd)
  if step % 500  == 0:
    cre = sess.run(cross_entropy, feed_dict=fd)
    acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
    print("step=", step, "cre=", cre, "acc=", acc)

# final rate
acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print("right rate=", acc)
