import tensorflow as tf

# define placeholder
a = tf.placeholder(tf.int32, [None])

# define an operation
b = tf.constant(10)
x10_op = a * b

# start a session
sess = tf.Session()

# git data to placeholder and run the session
result1 = sess.run(x10_op, feed_dict={a: [1,2,3,4,5]})
print(result1)
result2 = sess.run(x10_op, feed_dict={a: [10, 20]})
print(result2)
