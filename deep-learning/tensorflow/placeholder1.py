import tensorflow as tf

# define placeholder
a = tf.placeholder(tf.int32, [3])

# define a constant and an operation
b = tf.constant(2)
x2_op = a * b

# start a session
sess = tf.Session()

# give data to placeholder and run the session
result1 = sess.run(x2_op, feed_dict={ a: [1, 2, 3] })
print(result1)
result2 = sess.run(x2_op, feed_dict={ a: [10, 20, 30] })
print(result2)
