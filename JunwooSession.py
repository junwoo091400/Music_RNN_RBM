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

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())