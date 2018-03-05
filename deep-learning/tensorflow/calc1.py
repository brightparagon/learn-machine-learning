import tensorflow as tf

# define constants
a = tf.constant(1234)
b = tf.constant(5000)

add_op = a + b

# start a session
sess = tf.Session()
result = sess.run(add_op)
print(result)
