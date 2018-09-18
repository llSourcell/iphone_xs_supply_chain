# python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import packages
import numpy as np
import time
import os
from scipy.stats import poisson

# local import
from parameters import *

FLOAT_DATA_TYPE = np.float32

### can be better --> not a good coding style
pois_a_rent = poisson(RENT_EXPECT_A)
pois_a_return = poisson(RETURN_EXPECT_A)
pois_b_rent = poisson(RENT_EXPECT_B)
pois_b_return = poisson(RETURN_EXPECT_B)

# distribution of car rental in location a
def a_rent_prob(n):
	p = pois_a_rent.pmf(n)
	return p
# distribution of car return in location a
def a_return_prob(n):
	p = pois_a_return.pmf(n)
	return p
# distribution of car rent in location b
def b_rent_prob(n):
	p = pois_b_rent.pmf(n)
	return p
# distribution of car return in location b
def b_return_prob(n):
	p = pois_b_return.pmf(n)
	return p

class Model:
	def __init__(self, config):
		'''
			Input:
				config contains the following attribute
					N_a : possible choices of cars number in "a" location, 
						  e.g. if number of cars can vary from 0~20, then N_a = 21
					N_b : possible choices of cars number in "b" location
					N_move : maximum number of cars that can be moved
							 --> total action number = 2*N_move + 1 
					rent_price : revenue earned from renting a car
					move_cost : cost to move a car from a to b or from b to a
					gamma : discount factor in MDP
					P_all_filepath : file path of precomputed P_all matrix
					R_all_filepath : file path of precomputed R_all matrix
		'''
		self._N_move = config.N_move
		self._N_act = N_act = 2*self._N_move + 1
		self._N_a = N_a = config.N_a
		self._N_b = N_b = config.N_b
		self._max_storage = max(self._N_a, self._N_b) - 1
		self._rent_price = config.rent_price
		self._move_cost = config.move_cost
		self._gamma = config.gamma
		self._P_filepath = config.P_all_filepath
		self._R_filepath = config.R_all_filepath
		# transition matrix associated with triplet (action, state, next_state)
		self._P_all = np.zeros((N_act, N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
		# reward table associated with triplet (action, state, next_state)
		self._R_all = np.zeros((N_act, N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
		# transition matrix corresponding to certain policy
		self._P_this_policy = np.zeros((N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
		# reward table corresponding to certain policy
		self._R_this_policy = np.zeros((N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
		# value function of policy estimated currently
		self._V = np.zeros((N_a*N_b), dtype=FLOAT_DATA_TYPE)
		# policy table associated with 2-tuple (state, next_state)
		# initialize to uniform no action
		# self._policy = np.zeros((N_a*N_b), dtype=np.int8) ## alternative choice
		# initialize to random action -N_move~+N_move
		self._policy = np.random.randint(low=-(self._N_move), high=self._N_move, size=(N_a*N_b), dtype=np.int8)

	def train(self, tol, to_form_all=False):
		'''
			Input:
				tol : tolerance of error in policy evaluation
				to_form_all : True-->will form new P_all and R_all matrix
							  False-->load precomputed P_all and R_all matrix 
		'''
		print('start training')
		# form or load P_all and R_all
		if to_form_all:
			self.form_all()
			f_name = 'P_all_%d_%d' %(self._max_storage, self._N_move)
			np.save(f_name, self._P_all)
			f_name = 'R_all_%d_%d' %(self._max_storage, self._N_move)
			np.save(f_name, self._R_all)
			# np.save('P_all.npy', self._P_all) ## testing
			# np.save('R_all.npy', self._R_all) ## testing
		else:
			self._P_all = np.load(self._P_filepath)
			self._R_all = np.load(self._R_filepath)
			# self._P_all = np.load('P_all.npy') ## testing
			# self._R_all = np.load('R_all.npy') ## testing

		print('initial policy') # debug
		print(np.reshape(self._policy, (self._N_a,self._N_b))) ## debug

		error = 100
		n_iters = 0 # iteration
		while error != 0:
			n_iters = n_iters + 1 # iteration
			print('Iter%d:' %n_iters) # iteration
			# take one step --> policy evaluation and policy improvement
			new_V, new_policy = self.take_step(tol)
			# check difference between current policy and improved policy
			error = np.sum(np.absolute(new_policy-self._policy))
			# check for every state if value function with improved 
			# policy is better than that with old policy
			tmp = new_V - self._V
			if tmp[tmp<0].any():
				print('ERROR: value function do not improve for all states.')
				break
			# update policy and value function
			self._policy = new_policy
			self._V = new_V
			print(np.reshape(self._policy, (self._N_a,self._N_b))) ## debug
			print('--policy err = %d' %error)

		print('end training')

	def take_step(self, tol):
		# print('--take one step')
		# initialize value function V
		self.init_value_function()
		# get P_this_policy and R_this_policy
		self.get_this_policy_PR()
		### policy evaluation ###
		# print('----policy evaluation')
		error = 100
		while error > tol:
			# V_next is a matrix with all rows the same and each row equals to value function
			V_next = np.tile(self._V, (self._N_a*self._N_b,1))
			# update value function
			new_V = np.sum(self._P_this_policy*(self._R_this_policy+self._gamma*V_next), axis=1)
			# compute error
			error = np.sum(np.square(new_V-self._V)) # sum of square
			print('--value function err = %f' %error)
			# update value function
			self._V = new_V

		### greedy policy improvement ###
		# print('----policy improvement')
		score = np.zeros((self._N_act, self._N_a*self._N_b))
		V_next = np.tile(self._V, (self._N_a*self._N_b,1))
		for act in xrange(self._N_act):
			score[act] = np.sum(self._P_all[act]*(self._R_all[act]+self._gamma*V_next), axis=1)
		new_policy = np.argmax(score, axis=0)
		# from range 0~10 to -N_move~+N_move
		new_policy = new_policy - self._N_move

		return new_V, new_policy

	def init_value_function(self):
		### initialize value function in the beginning of policy evaluation
		# print('----initialize value function')
		self._V = self._V # start from previous computed value function

	def form_all(self):
		### form P_all and R_all, basic setup
		print('--form P_all and R_all')
		start = time.time() ### timer
		### can be better --> parallel programming
		tmp_P = np.zeros((self._N_act, self._N_a, self._N_b, self._N_a, self._N_b), dtype=FLOAT_DATA_TYPE)
		tmp_R = np.zeros((self._N_act, self._N_a, self._N_b, self._N_a, self._N_b), dtype=FLOAT_DATA_TYPE)
		for s_a in xrange(self._N_a):
			start3 = time.time() ### timer
			for s_b in xrange(self._N_b):
				# start2 = time.time() ### timer
				for s_a_next in xrange(self._N_a):
					# start1 = time.time() ### timer
					for s_b_next in xrange(self._N_b):
						for act in xrange(self._N_act):
							tmp_P[act, s_a, s_b, s_a_next, s_b_next], \
									tmp_R[act, s_a, s_b, s_a_next, s_b_next] \
											= self.compute(s_a, s_b, s_a_next, s_b_next, act)					# end1 = time.time() ### timer
					# print('*** s_a_next one step time: %f' %(end1-start1)) ### timer
				# end2 = time.time() ### timer
				# print('*** s_a_next one step time: %f' %(end2-start2)) ### timer
			end3= time.time() ### timer
			print('*** s_a %d one step time: %f' %(s_a, end3-start3))
		end = time.time() ### timer
		# N_state is N_a - 1
		# reshape from N_act*N_state*N_state*N_state*N_state to N_act*(N_state*N_state)*(N_state*N_state)
		self._P_all = np.reshape(tmp_P, self._P_all.shape)
		self._R_all = np.reshape(tmp_R, self._P_all.shape)
		print('****** elasped time in form_all %f' %(end-start))

	def get_this_policy_PR(self):
		### update P_this_policy and R_this_policy according to 
		### current policy and looking up P_all and R_all
		# print('----get P and R of current policy')
		# from range -N_move~+N_move to 0~N_act
		index = self._policy + self._N_move
		### can be better, parellel indexing
		for i in range(self._N_a*self._N_b):
			for j in range(self._N_a*self._N_b):
				self._P_this_policy[i, j] = self._P_all[index[i], i, j]
				self._R_this_policy[i, j] = self._R_all[index[i], i, j]


	def compute(self, s_a, s_b, s_a_next, s_b_next, act):
		### compute transition probabilty and expected immediate 
		### reward given this state, next state, and action
		act = act - self._N_move # from range 0~N_act to -N_move~+N_move, +: a-->b, -: b-->a
		# cars moved from one location cannot be more than number of cars in that location
		if (act>0 and act>s_a) or (act<0 and -1*act>s_b):
			return 0, 0
		# compute difference between cars number in current state(today) and next state(tommorrow)
		a_diff = s_a_next - (s_a-act)
		b_diff = s_b_next - (s_b+act)
		# maximum number of cars which can be rented, i.e. number of cars in one location after act
		a_max_rent = s_a - act
		b_max_rent = s_b + act
		# cannot surpass maximum storage
		if a_max_rent>self._max_storage or b_max_rent>self._max_storage:
			return 0, 0 
		### go through all possibility from s_a(today) to s_a_next(tommorrow) with act done overnight
		# in location a
		r_a = p_a = 0
		for a_rent in xrange(a_max_rent,-1,-1): # loop from a_max_rent to 0
			a_return = a_rent + a_diff
			# number of cars returned to location a is not allowed to be negative
			if a_return<0:
				break
			tmp = a_return_prob(a_return) * a_rent_prob(a_rent)
			r_a = r_a + (a_rent*self._rent_price) * tmp
			p_a = p_a + tmp
		# in location b
		r_b = p_b = 0
		for b_rent in xrange(b_max_rent,-1,-1): # loop from b_max_rent to 0
			b_return = b_rent + b_diff
			# number of cars returned to location a is not allowed to be negative
			if b_return<0:
				break
			tmp = b_return_prob(b_return) * b_rent_prob(b_rent)
			r_b = r_b + (b_rent*self._rent_price) * tmp
			p_b = p_b + tmp
		# compute total expected reward and transition possibility
		r = r_a + r_b - np.absolute(act)*self._move_cost
		p = p_a * p_b

		return p, r

	@property
	def policy(self):
		return self._policy

	@property
	def V(self):
		return self._V
	
