import tensorflow as tf

# define constants and variables
a = tf.constant(100, name="a")
b = tf.constant(200, name="b")
c = tf.constant(300, name="c")
v = tf.Variable(0, name="v")

# define data flow
calc_op = a + b * c
assign_op = tf.assign(v, calc_op)

# start a session
sess = tf.Session()

# use tensorboard
tw = tf.summary.FileWriter("log_dir", graph=sess.graph)

# run the session
sess.run(assign_op)
print(sess.run(v))
