import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation as mma
import RBM
import rnn_rbm_train as rrt

def main(target_dir):
	print('Generating Music from Music, just plain matrixifing,,, lol')
	print('You Entered : *',target_dir,'*')
	matrixified = mma.get_song(target_dir)
	exportDir = "music_outputs/{}".foramt(target_dir.split('/')[-1].split('.')[0])
	mma.write_song(exportDir,matrixified)

main(sys.argv[1])