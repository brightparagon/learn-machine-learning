import tensorflow as tf

# define constants
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)

# define operations
calc1_op = a + b * c
calc2_op = (a + b) * c

# start a session
sess = tf.Session()
result1 = sess.run(calc1_op)
print(result1)
result2 = sess.run(calc2_op)
print(result2)
