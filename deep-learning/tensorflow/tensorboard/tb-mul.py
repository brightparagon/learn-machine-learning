import tensorflow as tf

# define data flow
a = tf.constant(20, name="a")
b = tf.constant(30, name="b")
mul_op = a * b

# start a session
sess = tf.Session()

# write graph for tensorboard
tw = tf.summary.FileWriter("log_dir", graph=sess.graph)

# run the session
print(sess.run(mul_op))
