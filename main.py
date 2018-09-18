# python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import packages
import numpy as np
import argparse
import time
import os

# local import
from parameters import *
import Jack_Cars_Model

def main():
	# get configuration from parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_move', type=int, default=N_MOVE,
						help='Maximum number of car that can be moved, default 5')
	parser.add_argument('--N_a', type=int, default=N_A,
						help='Choices of cars number in location a, default 21 --> 0~20 cars')
	parser.add_argument('--N_b', type=int, default=N_B,
						help='Choices of cars number in location b, default 21 --> 0~20 cars')
	parser.add_argument('--rent_price', type=int, default=RENT_PRICE,
						help='Money earned from renting a car, default 4')
	parser.add_argument('--move_cost', type=int, default=MOVE_COST,
						help='Cost of moving one car from one place to the other, default 2')
	parser.add_argument('--gamma', type=float, default=GAMMA,
						help='Discount factor of MDP, default 0.9')
	parser.add_argument('--to_form_all', type=bool, default=False,
						help='True if you want to form new matrices associated with problem setting, default False')
	parser.add_argument('--P_all_filepath', type=str, default=DEFAULT_P_PATH,
						help='File path of precomputed P_all matrix, default dynamics/P_all_20_5.npy')
	parser.add_argument('--R_all_filepath', type=str, default=DEFAULT_R_PATH,
						help='File path of precomputed R_all matrix, default dynamics/R_all_20_5.npy')
	config = parser.parse_args()

	model = Jack_Cars_Model.Model(config)
	model.train(POLICY_EVAL_TOL, to_form_all=config.to_form_all)
	optim_policy = model.policy 
	optim_policy = np.reshape(optim_policy, (N_A,N_B))
	# print('optimal policy:')
	# print(optim_policy)

if __name__=='__main__':
	main()