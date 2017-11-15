import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation 

import ipdb

import os

"""
	This file contains the code for training the RNN-RBM by using the data in the Pop_Music_Midi directory
"""


batch_size = 100 #The number of trianing examples to feed into the rnn_rbm at a time
epochs_to_save = 10#The number of epochs to run between saving each checkpoint
saved_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file

k_list = [5,10,15]

def main(num_epochs, k_test):
	#num_epochs = 100 # 100! (9, 19, 29, ... 99 [10 Checkpoints.])
	target_dir = 'Train_DATA'
	#First, we build the model and get pointers to the model parameters
	songs = midi_manipulation.get_songs(target_dir) #Load the songs 

#######################
	song_primer = []
	primer_song = ['You Belong With Me - Verse.midi', 'Someone Like You - Chorus.midi', 'Pompeii - Bridge.midi']
	primer_song = [os.path.join(target_dir,p) for p in primer_song]
	song_primer = [ midi_manipulation.get_song(primer_song[i]) for i in range(3) ]
#######################

	#ipdb.set_trace()
	print('Doing K as:',k_test)
	x, out1, out2, cost, generate, W, bh, bv, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm(k_test)

	#The trainable variables include the weights and biases of the RNN and the RBM, as well as the initial state of the RNN
	tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0] 
	# opt_func = tf.train.AdamOptimizer(learning_rate=lr) 
	# grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1)
	# updt = opt_func.apply_gradients(zip(grads, tvars)) 
	
	#The learning rate of the  optimizer is a parameter that we set on a schedule during training
	opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
	gvs = opt_func.compute_gradients(cost, tvars)
	gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs] #We use gradient clipping to prevent gradients from blowing up during training
	updt = opt_func.apply_gradients(gvs)#The update step involves applying the clipped gradients to the model parameters

	saver = tf.train.Saver(tvars, max_to_keep=None) #We use this saver object to restore the weights of the model and save the weights every few epochs

	loss_print_dir = 'k{}_lossprint.csv'.format(k_test)
	Loss_Print_pipe = open(loss_print_dir,'w')

	def Generate_Music(k_test,epoch):
		for i in tqdm(range(3)):
			generated_music = sess.run(generate(300), feed_dict={x: song_primer[i]}) #Prime the network with song primer and generate an original song
			new_song_path = "music_outputs/k{}_e{}_{}".format(k_test, epoch, primer_song[i].split('/')[-1].split('.')[0]) #The new song will be saved here
			midi_manipulation.write_song(new_song_path, generated_music)

	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init) 
		#os.system("python weight_initializations.py Train_DATA")

		saver.restore(sess, saved_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

		print("First, we print these songs as they are. Natural Baby!")
		for i in range(3):
			original_song_path = "music_outputs/{}".format(primer_song[i].split('/')[-1].split('.')[0])
			midi_manipulation.write_song(original_song_path, song_primer[i])

		#We run through all of the songs n_epoch times
		print "starting"

		for epoch in range(num_epochs):
			costs = []
			start = time.time()
			for s_ind, song in enumerate(songs):
				for i in range(0, len(song), batch_size):
					tr_x = song[i:i + batch_size] 
					#alpha = min(0.01, 0.1/float(i)+0.001) #We decrease the learning rate according to a schedule.
					alpha = 0.01
					_, out_1, out_2, C = sess.run([updt, out1, out2, cost], feed_dict={x: tr_x, lr: alpha}) 
					costs.append(C) 
			#Print the progress at epoch
			out_1 = np.mean(out_1)
			out_2 = np.mean(out_2)
			if Loss_Print_pipe.closed == False:
				Loss_Print_pipe.write("{},{},{},{},{}\n".format(epoch, out_1, out_2 ,np.mean(costs), time.time()-start))
			#ipdb.set_trace()
			print "epoch: {} out1: {} out2:{} cost: {} time: {}".format(epoch, out_1, out_2 ,np.mean(costs), time.time()-start)
			print
			#Here we save the weights of the model every few epochs
			if (epoch + 1) % epochs_to_save == 0: 
				saver.save(sess, "parameter_checkpoints/k{}_epoch_{}.ckpt".format(k_test,epoch))
				Generate_Music(k_test,epoch)
	Loss_Print_pipe.close()#Close Exporing Pipe.

main(int(sys.argv[1]),int(sys.argv[2]))