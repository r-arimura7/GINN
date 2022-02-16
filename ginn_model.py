# GINN emulator; Oct/26/21 - Dec/2/21

from re import I
from numpy.lib.function_base import append
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.eager.def_function import run_functions_eagerly
#from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.keras.backend import dtype
#from tensorflow.python.keras.engine import data_adapter
import pickle
#from tensorflow.keras.backend import eval
import random
import os
# tf.compat.v1.disable_eager_execution()

class Model_wrapper(object):
	def __init__(self, model):
		self.model = model

class GINN_inputLayer(layers.Layer):
	def __init__(self, Weights, units, il_batch_input_shape):
		#W is numpy array from imported pickel.
		super(GINN_inputLayer, self).__init__(dynamic= True,batch_input_shape = il_batch_input_shape) #instantiate super class. 
		print('is eager in __init__',tf.executing_eagerly())
		self.Weights = Weights
		self.K = len(Weights) # number of clusters (or concepts)
		self.units = units
		print('il_batch_input_shape is ',il_batch_input_shape)
		self.batch_size =  il_batch_input_shape[0]
		print('self.batch_size is ',self.batch_size)
		self.n = np.zeros(self.K, dtype=int)
		self.m = 0 # Total number of words`
		# self.n[k] is the number fo words contained in the k'th concept
		for k in range(self.K):
			n_k = self.Weights[k].shape[1]
			self.n[k] = n_k
			self.m += n_k

	def build(self, input_shape):
		self.processed_W = [k[0].flatten() for k in self.Weights]
		self.flattened_W = np.concatenate(self.processed_W)
		self.flattened_W_tfv = tf.Variable(self.flattened_W, trainable=True, dtype='float32')
		# for w in self.Weights: # register trainable variables
		# 	self.W.append(tf.Variable(w.tolist(), trainable=True, dtype='float32'))
		# init_array =  np.zeros((2,self.K))
		# self.H_star_j_t =tf.Variable(init_array,dtype='float32',trainable=False)
		super(GINN_inputLayer, self).build(input_shape)
		print('is eager in build',tf.executing_eagerly())

	def call(self, inputs):
		print('inputs in call is ',inputs)
		print('is eager in call',tf.executing_eagerly())
		print('self.K is ',self.K)
		print('self.n is ',self.n)
		print('self.flattened_W is ',self.flattened_W)
		return self.GINN_op(inputs)

	@tf.custom_gradient
	def GINN_op(self,x):
		# x is frequencies, K is number of cluster, n is a container includes n_k, falltened_W is tf.Variables) 
		# Creating forward pass
		self.Z = []
		self.u2 = []
		self.vCS = []
		#DEBUG now.
		for j in range(self.batch_size): #x.shape[0]represents batch size, be consistent!
			xs = x[j][0][:]
			print('xs is ',xs)
			ws = []
			z_j = []
			k0 = 0
			print('is eager in custom_gradient',tf.executing_eagerly())
			# print('K is', K)
			for k in range(self.K):
				# is same as self.K which represents number of cluster.
				k1 = k0 + self.n[k]
				z_k = xs[k0:k1]
				z_j.append(z_k)
				weight_vector = tf.expand_dims(self.flattened_W_tfv[k0:k1],axis=0)
				ws.append(tf.linalg.matvec(weight_vector, z_k))
				k0 = k1
			self.Z.append(z_j)
			u2_j = tf.concat(ws,axis=0) # you can use this in algo1()
			print('u2_j =', u2_j)
			self.u2.append(u2_j)
			vCS_j = tf.keras.activations.tanh(u2_j)
			print('vCS_j is',vCS_j)
			self.vCS.append(vCS_j)
			print('self.vCS is ',self.vCS)
		# Creating backward pass.
		def grad_GINN_op(*upstream, variables = self.flattened_W_tfv):# 
			grad_xs = [0] # very stub!
			#print('grad_xs is ',grad_xs)
			dy_dws = []
			grad_vars = []  # To store gradients of passed variables
			print('*upstream right before asseritons are',*upstream)
			print('variables right before asseritons are',variables)
			for k in range(self.K):
				dy_dw =  self.algo1(k) #change variables[k]　to self.w[k]
				dy_dws.append(dy_dw)
				#fallaten dy_dws to store into grad_vars
			intermediate_grad_vars = [item for i in dy_dws for item in i]
			grad_vars = tf.expand_dims(intermediate_grad_vars,axis=0)
			return grad_xs, grad_vars
		return self.vCS, grad_GINN_op
	
	def algo1(self, k): #Implementing Algorithm 1 of Ito et al.(2020), pp.434
		print('YOU ARE IN ALGO 1')
		print('is eager in algo1',tf.executing_eagerly())
		# update self.W[k] by reading document j whose polarity is b_j 
		# j : document number
		# k : cluster number
		# m := sum_{k} n(k)
		# i : word number in the cluster, the word is represented by w_{k,i}
		# z^k_{j,i} : a real number
		# z^k_j := (z^k_{j,1}, ... z^k_{j,n(k)) (1 x n(k))
		sum_of_prod_of_delta_k_2_star_and_z_k = 0 #sum or product of delta_k_2_star_and z_k ,i.e., frequency, of doc j. See line 17 of Algorithm1
		for j in range(self.batch_size): #self.batch_size == the cardinality of Omega_m
			# data = InputData() #stub 
			d_j = self.transform_label_to_matrix(self.data.labels[j])
			print(d_j)
			# assert type(d_j) == numpy.ndarray 
			y_j = self.y[j] # minibatch dimension already being considered here.
			y_j_T = y_j.numpy()
			print('y_j_T is',y_j_T)
			delta_j_4 = np.subtract(y_j_T,d_j)
			# print('delta_j_4 is ',delta_j_4)
			self.w3 = self.model.model.layers[1].get_weights()[0]
			# print(tf.executing_eagerly())
			self.w4 = self.model.model.layers[2].get_weights()[0]
			# print('w3 is ', self.w3.shape, 'vCS.numpy() is ', self.vCS.numpy().shape)
			vCS_j = self.vCS[j].numpy()
			u3_j = np.matmul(self.w3.T, vCS_j)
			# print('u3_j', u3_j)
			derivative_f3 = self.tanh_derivative(u3_j)
			diagonailized_var = np.diag(derivative_f3) 
			# print('diagonailized_var.shape is ',diagonailized_var.shape)
			matmuled_var = np.matmul(diagonailized_var, self.w3.T)
			H_j_t = np.matmul(self.w4.T, matmuled_var)
			
			try: 
				H_star_j_t
			except UnboundLocalError:
				H_star_j_t = np.zeros((2,self.K),dtype=float)

			for local_k in range(self.K):
				if H_j_t[0][local_k] < 0:
					H_star_j_t[0][local_k] = H_j_t[0][local_k]
				if H_j_t[1][local_k] > 0:
					H_star_j_t[1][local_k] = H_j_t[0][local_k]
				
			RHS_of_delta_j_2 = np.matmul(H_star_j_t.T, delta_j_4.T)
			
			u2_j_np= self.u2[j].numpy()
			u2_j = np.expand_dims(u2_j_np,axis=1)
			# print('np.tanh(u2_j)**2 is ',np.tanh(u2_j)**2)
			# print('(np.tanh(u2_j)**2).shape is ',(np.tanh(u2_j)**2).shape)
			LHS_of_delta_j_2 = 1- (np.tanh(u2_j))**2
			delta_2_j = np.multiply(LHS_of_delta_j_2,RHS_of_delta_j_2) # line 15 of Algorithm 1 in Ito et al.(2020), pp.434 
			sum_of_prod_of_delta_k_2_star_and_z_k += delta_2_j[k] * self.Z[j][k] #z_j = self.z[j]
		print('sum_of_var is ',sum_of_prod_of_delta_k_2_star_and_z_k)

		grad_wk2 = sum_of_prod_of_delta_k_2_star_and_z_k[k] / self.batch_size # Denominator is stub. Intention is to take average by Total N.
		
		"""
		# necesary values: inputlayer:u2_j, v_j,  y_j^CS, label, W3, 
		grad_wk2 = np.array([0.005]) # stub
		"""
		# necesary values: inputlayer:u2_j, v_j,  y_j^CS, label, W3, 
		# for j in range(self.Omega_m):
			# d_j =  (0,1)^T if label_j = true; otherwise: (1,0)^T 
			# u_j2 = u2_j # = tanh^{-1}(v_j^CS)
			# u_j2_grad = 1 - (u_j2 ^2)
			# u_j3 = W3 * v_j^CS # stub
			# delata_j4 = y_j - d_j  # y_j := softmax(W4 * tanh(W3 * v_j^S) + b0) # stub
			# H_jt = W4 * diag(u_j2_grad(u_j3)) * W3 # \in R^{2 x K)
			# ...
		# """
		return grad_wk2

	# def get_weights_from_model_wrapper(self,modelwrapper):
	# 	rcvd_weights = modelwrapper.model.layers[1].get_weights()[0]
		# rcvd_layers = modelwrapper.model.layers[1].get_weights()[0]# this will get error:
		# Cannot get value inside Tensorflow graph function.
 
		return rcvd_weights
		# return self.weights 
	#def create_H_star_j_t(self,H_j_t,H_star_j_t):
	#	print('funciton THURU')
	#	func1 = tf.add(4,5) 
	#	return func1() 

	def tanh_derivative(self, x):
		y = 4/(np.exp(x)+np.exp(-x))**2
		return y	

	def transform_label_to_matrix(self, datalabel):
		if datalabel[0]== 1:
			d_j = np.array([0,1]).T #document is positive. 
			# d_j = d_j.set_shape([2,1]) 
			# print('d_j is positive')
		elif datalabel[0] == 0:
			d_j = np.array([1,0]).T #document is negative.
			# print('d_j is negative')
		else:
			print('Error:label data given but the label is neither 1 nor 0.') 
		return d_j 
	
	def extract_vars(self, model):
		# extract variables that will be used in algo1() from other layers
		#self.model = model # this led an infinite recursion.:w
		self.model = Model_wrapper(model)
		self.data = self.model.model.data
	
	def set_y(self, y):
		# print('y is', y)
		# print('tf.executing_eagerly in the function T or F',tf.executing_eagerly())
		self.y = y
		# print('Post config y is',y)
	

class GINN_model(keras.Model):
	def __init__(self, data):
		super(GINN_model, self).__init__()
		self.data = data
	
	def build(self, input_shape):
		print('is eager in GINN_model build',tf.executing_eagerly())
		self.inputlayer = GINN_inputLayer(self.data.W,units = self.data.inputs,il_batch_input_shape= (10,1,915))#stub
		self.K = self.inputlayer.K
		self.K2 = self.K*2 # stub AR COMMENT: two edges from Concept layer to Context. See Fig 1 of Ito et al.(2020)
		self.secondlayer = layers.Dense(self.K2, activation="tanh",kernel_initializer='random_normal',use_bias = False)
		# print('initalized secondlyaer is', self.secondlayer.weights)
		self.outputlayer = layers.Dense(2, activation='softmax') #stub 10 is cardinality of minibatch.
		self.inputlayer.extract_vars(self)
		print('input_shape is ',input_shape)
		super(GINN_model, self).build(input_shape)
	
	
	def call(self, inputs): # inputs = v^{BOW}_j
		# print('model.input.shape=', inputs.shape)
		# print('model.input[1]=', inputs[1])
		# print('model.input[1][0]=', inputs[1][0])
		# print('type of model.input=', type(inputs))
		print('inputs',inputs)
		print('is eager in GINN_model call',tf.executing_eagerly())
		vCS = self.inputlayer(inputs) # very stub
		# vCS = tf.expand_dims(vCS,axis=0)
		print('pre-concatenation vCS is ',vCS)
		# vCS = tf.concat(vCS,axis= 0 )
		# print('post-concatenation vCS is ' ,vCS)
		vCS = tf.expand_dims(vCS,axis=1)
		print('post-expand_dims vCS is ' ,vCS)
		V3 = self.secondlayer(vCS)
		self.inputlayer.w3 = self.secondlayer.weights
		print('V3 is ',V3)
		y = self.outputlayer(V3)
		print('y is ',y)
		self.y = y
		print('self.y in GINN_model call() is',self.y)
		print('self.y.shape in GINN_model call() is',self.y.shape)
		# print('tf.executing_eagerly T or F',tf.executing_eagerly())
		# tf.config.run_functions_eagerly(True)
		self.inputlayer.set_y(self.y)
		y_prob_pred = tf.math.reduce_max(y, axis=2,keepdims= True)
		print('y_prob_pred is ',y_prob_pred)

		# stub_pred_of_y = tf.squeeze(tf.slice(self.y,[0,0,0],[1,10,1]),axis=0) #very stub as pred is chosen from manual slicing. Pred shoud be selected from argamaxed-index!!	
		# print('stub_pred_of_y is ',stub_pred_of_y )
		return y_prob_pred 



class InputData(object):
	def __init__(self):
		self.read_pickles()
		self.preprocess_input()
	
	def read_pickles(self):
		#importing wegiht 
		with open('data/W.pkl', 'rb') as fin:
			self.W = pickle.load(fin)
		#importing training_data(i.e. vbow) 
		with open('data/training_data.pkl', 'rb') as fin:
			self.training_data = pickle.load(fin)
		#print('len(training_data)=', len(self.training_data)) # =56
		t0 = self.training_data[0]
		#importing label (positive = 1, negative = 0) 
		with open('data/labels.pkl', 'rb') as fin:
			self.labels = pickle.load(fin)
		#print('len(labels)=',len(self.labels)) # =56
		#print('labels=',self.labels)
		
		print('===End Reading Pickles==')
	
	def preprocess_input(self):
		"""
		change numpy ndarray training_data AND labels to tf.data.Dataset data.
		"""
		#1.Preprocess training data
		preprocessed_training_data = [x for x in self.training_data ]
		data_intermediatelist = []
		for cnt in range(len(preprocessed_training_data)):
			di = preprocessed_training_data[cnt][:].tolist()
			#data_intermediatelist.append(di)
			data_intermediatelist.append(sum(di,[])) # flatten
		#print('data_intermediatelist[0]=', data_intermediatelist[0])
		data_frequencies = tf.expand_dims(tf.constant(data_intermediatelist,dtype = 'float32'),axis= 1) #expand dim at axis 1 to enable future propergation among layers.

		self.x = data_frequencies
		print('self.x is ',self.x)
		# self.x = tf.expand_dims(self.x,axis =2) #Adding dummy dimension for future preprocessing spefically fro Binary cross entropy and its argument 'reduction=tf.keras.losses.Reduction.NONE'.
		print('self.x is ',self.x)
		
		#self.frequencies = tf.ragged.constant(self.preprocessed_input_data, dtype='float32')	

		#2.Preprocess label
		self.formatted_labels=tf.expand_dims(tf.constant(self.labels[:].tolist(),dtype= 'float32'),axis=1)
		print(self.x)
		print(self.formatted_labels)
		#print('self.formatted_labels=', self.formatted_labels)
		self.inputs = tf.data.Dataset.from_tensor_slices((self.x, self.formatted_labels)).batch(10
		,drop_remainder= True)
		print('top.input=', self.inputs)

#ToDo write decorator to save output to .txt file
def output_to_txt_file(f):
	def log():
		# os.system('script test.txt')
		print('start now')
		f()

	return log 
	

# @output_to_txt_file
def main():
	data = InputData()
	g_model = GINN_model(data)
	g_model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(),run_eagerly = True) # you need 'run_eagerly = True' arg to run the whole process in eager mode.
	# print('data.inputs is ',data.inputs)
	# print('type of data.inputs is ',type(data.inputs))
	print(g_model.run_eagerly)
	g_model.fit(data.inputs, epochs=3 )
	g_model.summary()
	# g_model.model.layers[1].get_weights()[0]
	# print('g_model.secondlayer',g_model.secondlayer.get_weights()[0])
	# test_weights = g_model.secondlayer.weights
	# print(test_weights) 
	# run test 
	#result= g_model.predict(sample_x)
	#print(result)

main()