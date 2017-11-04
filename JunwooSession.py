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

bh  = tf.Variable([[1.,2.,3.,4.,5.]])

y = (tf.matmul(x,W) + bh)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict = {x:[[5.],[6.]]})


print result
