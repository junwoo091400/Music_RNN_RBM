import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation as mma
import RBM
import rnn_rbm_train as rrt

import midi

def main(model,k_gen):
	target_dir = 'Train_DATA'
	#First, we build the model and get pointers to the model parameters
	songs = midi_manipulation.get_songs(target_dir) #Load the songs 
	song_primer = []
	primer_song = ['You Belong With Me - Verse.midi', 'Someone Like You - Chorus.midi', 'Pompeii - Bridge.midi']
	primer_song = [os.path.join(target_dir,p) for p in primer_song]
	song_primer = [ midi_manipulation.get_song(primer_song[i]) for i in range(3) ]
	x, out1, out2, cost, generate, W, bh, bv, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()


if __name__ == '__main__':
	main(sys.argv[1],int(sys.argv[2]))

'''
lowerBound = 24 #The lowest note
upperBound = 102 #The highest note
span = upperBound-lowerBound #The note range
num_timesteps      = 5 #The number of note timesteps that we produce with each RBM

def write_song(path, song,chan):
	#Reshape the song into a format that midi_manipulation can understand, and then write the song to disk
	song = np.reshape(song, (song.shape[0]*num_timesteps, 2*span))
	noteStateMatrixToMidi(song,chan=chan, name=path,)

def noteStateMatrixToMidi(statematrix,chan, name="example", span=span):
	statematrix = np.array(statematrix)
	if not len(statematrix.shape) == 3:
		statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
	statematrix = np.asarray(statematrix)
	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)
	
	span = upperBound-lowerBound
	tickscale = 55
	
	midi.ProgramChangeEvent(tick=0, channel=chan, data=[33])

	lastcmdtime = 0
	prevstate = [[0,0] for x in range(span)]
	for time, state in enumerate(statematrix + [prevstate[:]]):  
		offNotes = []
		onNotes = []
		for i in range(span):
			n = state[i]
			p = prevstate[i]
			if p[0] == 1:
				if n[0] == 0:
					offNotes.append(i)
				elif n[1] == 1:
					offNotes.append(i)
					onNotes.append(i)
			elif n[0] == 1:
				onNotes.append(i)
		for note in offNotes:
			track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
			lastcmdtime = time
		for note in onNotes:
			track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
			lastcmdtime = time
			
		prevstate = state
	
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)

	midi.write_midifile("{}.mid".format(name), pattern)


def main(target_dir,chan):
	print('Generating Music from Music, just plain matrixifing,,, lol')
	print('You Entered : *',target_dir,'*')
	matrixified = mma.get_song(target_dir)
	exportDir = "music_outputs/chan{}_{}".format(chan,target_dir.split('/')[-1].split('.')[0])
	write_song(exportDir,matrixified,chan)

main(sys.argv[1],int(sys.argv[2]))
'''