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

with tf.device("/gpu:0"):
  A = tf.random_normal([matrix_size, matrix_size])
  B = tf.random_normal([matrix_size, matrix_size])
  C = tf.matmul(A, B)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  t1 = datetime.datetime.now()
  sess.run(C)
  t2 = datetime.datetime.now()