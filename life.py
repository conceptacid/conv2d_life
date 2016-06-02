import numpy as np
import tensorflow as tf
import matplotlib.animation as animation
from matplotlib import pyplot as plt


W = 30
H = 30

#state0 = tf.ones([H,W]) #tf.random_uniform([H,W], maxval=2, dtype=tf.int32)
#state0 = tf.random_uniform([H,W], maxval=2, dtype=tf.int32)
npstate0 = np.zeros([H,W])
npstate0[2,5] = 1
npstate0[3,5] = 1
npstate0[4,5] = 1
state0 = tf.convert_to_tensor(npstate0)
state = tf.Variable(tf.cast(tf.reshape(state0,[1,H,W,1]), tf.float32))
kernel = tf.reshape(tf.ones([3,3]), [3,3,1,1])
neighbours = tf.nn.conv2d(state, kernel, [1,1,1,1], "SAME") - state
survive = tf.logical_and( tf.equal(state, 1), tf.equal(neighbours, 2))
born = tf.equal(neighbours, 3)
newstate = tf.cast(tf.logical_or(survive, born), tf.float32)

init = tf.initialize_all_variables()



fig = plt.figure()
with tf.Session() as sess:
	sess.run(init)
	sess.run(newstate)
	newstate_ = sess.run(tf.reshape(newstate, [H,W]))
	sess.run(tf.assign(state, newstate))
	plot = plt.imshow(newstate_, cmap='Greys', interpolation='nearest')

	def animateFn(num, sess, state, newstate):
		sess.run(newstate)
		newstate_ = sess.run(tf.reshape(newstate, [H,W]))
		sess.run(tf.assign(state, newstate))
		plot.set_array(newstate_)
		return plot

	ani = animation.FuncAnimation(fig, animateFn, 5, fargs=(sess, state, newstate), interval=2, blit=False)
	plt.show()
	#print sess.run(state0)
	#for i in range(2):
	#	newstate_ = sess.run(tf.reshape(newstate, [H,W]))
	#print newstate_
	#	plt.imshow(newstate_, cmap='Greys', interpolation='nearest')
	#	plt.show()

