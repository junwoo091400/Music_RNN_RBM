import os
import sys

k_list = [10,20,30,40,50]
def main(num_epoch):
	for k in k_list:
		os.system('python Junwoo_k_TEST.py {} {}'.format(num_epoch,k))
main(int(sys.argv[1]))