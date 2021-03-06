import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import RBM
import rnn_rbm
import time
import midi_manipulation
import random

"""
	This file contains the code for running a tensorflow session to generate music
"""


num = 3 #The number of songs to generate

def main(saved_weights_path,target_dir,kval):
	if(os.path.isdir(target_dir)):
		songsList = os.listdir(target_dir)
		randomSong = songsList[random.randint(0,len(songsList)-1)]
		primer_song = os.path.join(target_dir, randomSong) #The path to the song to use to prime the network
	else:#Specific Song!
		primer_song = target_dir
		
	print('Primer Song = ',primer_song)

	#This function takes as input the path to the weights of the network
	x,_,_, cost, generate, W, bh, bv, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()#First we build and get the parameters odf the network

	tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

	saver = tf.train.Saver(tvars) #We use this saver object to restore the weights of the model

	song_primer = midi_manipulation.get_song(primer_song) 

	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)
		saver.restore(sess, saved_weights_path) #load the saved weights of the network
		# Generate songs
		generated_music = sess.run(generate(300, k_in = kval), feed_dict={x: song_primer}) #Prime the network with song primer and generate an original song
		
		saved_weight_name = saved_weights_path.split('/')[-1].split('.')[0]
		primer_song_name = primer_song.split('/')[-1].split('.')[0]
		
		new_song_path = "music_outputs/Name={}_K={}_{}".format(saved_weight_name,kval,primer_song_name) #The new song will be saved here
		midi_manipulation.write_song(new_song_path, generated_music)

if __name__ == "__main__":
	main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
	
