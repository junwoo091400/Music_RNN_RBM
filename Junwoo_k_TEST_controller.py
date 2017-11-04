import os
import sys

k_list = [5,10,15]
def main(num_epoch):
	for k in k_list:
		os.system('Junwoo_k_TEST.py {} {}'.format(num_epoch,k))
main(int(sys.argv[1]))