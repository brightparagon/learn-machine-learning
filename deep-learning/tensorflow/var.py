import tensorflow as tf

# define constants
a = tf.constant(120, name="a")
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")

# define a variable
v = tf.Variable(0, name="v")

# define data flow
calc_op = a + b + c
assgin_op = tf.assign(v, calc_op)

# start a session
sess = tf.Session()
sess.run(assgin_op)

# print variable v
print(sess.run(v))
