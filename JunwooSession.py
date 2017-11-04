import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation as mma
import RBM
import rnn_rbm_train as rrt
POP = 'Pop_Music_Midi'

n_visible = 1
n_hidden = 5

 W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
 x  = tf.placeholder(tf.float32, [2, n_visible])

 bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")

 x = tf.constant([ [1.0,2.0,3.0] ])
w = tf.constant([ [2.0],[2.0],[2.0] ])
y = tf.matmul(x,w)
print x.get_shape()


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)


print result
