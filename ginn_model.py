# GINN emulator; Oct/26/21 - Dec/2/21

from re import I
from numpy.lib.function_base import append
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from tensorflow.python.eager.def_function import run_functions_eagerly
#from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.keras.backend import dtype
#from tensorflow.python.keras.engine import data_adapter
import pickle
import tensorflow_addons as tfa
#from tensorflow.keras.backend import eval
# tf.compat.v1.disable_eager_execution()

class Model_wrapper(object):
	def __init__(self, model):
		self.model = model

class GINN_inputLayer(layers.Layer):
	def __init__(self, Weights, units = None, il_batch_input_shape = None):
		#W is numpy array from imported pickel.
		super(GINN_inputLayer, self).__init__(dynamic= True, batch_input_shape = il_batch_input_shape) #instantiate super class. 
		# print('is eager in __init__',tf.executing_eagerly())
		self.Weights = Weights
		self.K = len(Weights) # number of clusters (or concepts)
		self.units = units
		# print('il_batch_input_shape is ',il_batch_input_shape)
		self.batch_size =  il_batch_input_shape[0]
		# print('self.batch_size is ',self.batch_size)
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
		self.flattened_W_tfv = [tf.Variable(i,trainable=True,dtype='float32') for i in self.flattened_W] 
		self.j_first = 0
		self.j_last = self.batch_size

		super(GINN_inputLayer, self).build(input_shape)

	def call(self, inputs):
		# print('is eager in call',tf.executing_eagerly())
		self.vCS = []
		self.Z = [] # moved here; taka, Mar/10
		self.u2 = [] # moved here; taka, Mar/10
		# if self.itertaion_num == []:
		# 	self.j_first = 0
		# 	self.j_last = self.batch_size

		for j in range(self.j_first,self.j_last): 
			All_data = self.data.data_frequencies_All_data
			intermediate_xs = All_data[j] #STUB! YOU MUST TAKE BATCH DIMENSION INTO ACCOUNT!
			xs = [float(x) for x in intermediate_xs]
			params = xs
			vCS = self.GINN_op(*params)
			self.vCS.append(vCS)
		if not self.j_last == self.data.lastrow_num_of_data:
			self.j_first = self.j_last
			self.j_last += self.batch_size
		elif self.j_last == self.data.lastrow_num_of_data:
			self.j_first = 0
			self.j_last = self.batch_size
			# pass #stub.. why does this work in the first place 2022/3/21
		return self.vCS 

	@tf.custom_gradient
	def GINN_op(self,*x):
		# x is frequencies, K is number of cluster, n is a container includes n_k, falltened_W is tf.Variables) 
		# Creating forward pass
		# print('is eager in custom_gradient',tf.executing_eagerly())
		xs = x 
		ws = []
		z_j = []
		k0 = 0
		for k in range(self.K):
			k1 = k0 + self.n[k]
			z_k = xs[k0:k1]
			z_j.append(z_k)
			weight_vector = tf.expand_dims(self.flattened_W_tfv[k0:k1],axis=0)
			ws.append(tf.linalg.matvec(weight_vector, z_k))
			k0 = k1
		self.Z.append(z_j)
		u2_j = tf.concat(ws,axis=0) # you can use this in algo1()
		self.u2.append(u2_j)
		vCS_j = tf.keras.activations.tanh(u2_j)
		# Creating backward pass.
		def grad_GINN_op(*upstream, variables = [self.flattened_W_tfv]):# 
			grad_xs = [tf.constant(1,dtype='float32') for _ in range(len(variables))]
			dy_dws = []
			grad_vars = []  # To store gradients of passed variables
			for k in range(self.K):
				dy_dw =  self.algo1(k) #change variables[k]　to self.w[k]
				dy_dws.append(dy_dw)
			#fallaten dy_dws to store into grad_vars
			intermediate_grad_vars = [tf.constant(item, dtype='float32') for i in dy_dws for item in i]
			grad_vars = intermediate_grad_vars 
			return grad_xs, grad_vars
		return vCS_j, grad_GINN_op
	def algo1(self, k): #Implementing Algorithm 1 of Ito et al.(2020), pp.434
		# print('is eager in algo1',tf.executing_eagerly())
		# update self.W[k] by reading document j whose polarity is b_j 
		# j : document number
		# k : cluster number
		# m := sum_{k} n(k)
		# i : word number in the cluster, the word is represented by w_{k,i}
		# z^k_{j,i} : a real number
		# z^k_j := (z^k_{j,1}, ... z^k_{j,n(k)) (1 x n(k))
		sum_of_prod_of_delta_k_2_star_and_z_k = 0 #sum of product of delta_k_2_star_and z_k ,i.e., frequency, of doc j. See line 17 of Algorithm1
		for j in range(self.batch_size): #self.batch_size == the cardinality of Omega_m
			d_j = self.transform_label_to_matrix(self.data.labels_train[j]) #RECONSIDER this... 
			# assert type(d_j) == numpy.ndarray 
			y_j = self.y[j] # minibatch dimension already being considered here.
			y_j_np = y_j.numpy()
			y_j_T = np.squeeze(y_j_np,axis=0) #Change matrix to vector to be consistent with Algo1 notation.
			delta_j_4 = np.subtract(y_j_T,d_j)
			self.w3 = self.model.model.layers[1].get_weights()[0]
			# print(tf.executing_eagerly())
			self.w4 = self.model.model.layers[2].get_weights()[0]
			vCS_j = self.vCS[j].numpy()
			u3_j = np.matmul(self.w3.T, vCS_j)
			derivative_f3 = self.tanh_derivative(u3_j)
			diagonailized_var = np.diag(derivative_f3) 
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
					H_star_j_t[1][local_k] = H_j_t[1][local_k]
				
			RHS_of_delta_j_2 = np.matmul(H_star_j_t.T, delta_j_4)
			u2_j= self.u2[j].numpy()
			LHS_of_delta_j_2 = 1- (np.tanh(u2_j))**2
			delta_2_j = np.multiply(LHS_of_delta_j_2,RHS_of_delta_j_2) # line 15 of Algorithm 1 in Ito et al.(2020), pp.434 
			sum_of_prod_of_delta_k_2_star_and_z_k += np.multiply(delta_2_j[k], self.Z[j][k]) #z_j = self.z[j]
		grad_wk2 = sum_of_prod_of_delta_k_2_star_and_z_k / self.batch_size 
		return grad_wk2

	def tanh_derivative(self, x):
		y = 1-(np.tanh(x))**2
		return y	

	def transform_label_to_matrix(self, datalabel):
		if datalabel[0]== 1:
			d_j = np.array([0,1]).T #document is positive. 
		elif datalabel[0] == 0:
			d_j = np.array([1,0]).T #document is negative.
		else:
			print('Error:label data given but the label is neither 1 nor 0.') 
		return d_j 
	
	def extract_vars(self, model):
		# extract variables that will be used in algo1() from other layers
		#self.model = model # this led an infinite recursion.:w
		self.model = Model_wrapper(model)
		self.data = self.model.model.data
	
	def set_y(self, y):
		# print('tf.executing_eagerly in the function T or F',tf.executing_eagerly())
		self.y = y
	

class GINN_model(keras.Model):
	def __init__(self, data):
		super(GINN_model, self).__init__()
		self.data = data
		self.data_d0 = self.data.batch_size
		self.data_d1 = self.data.row_dimension
		self.data_d2 = self.data.feature_dimension
	
	def build(self, input_shape):
		# print('is eager in GINN_model build',tf.executing_eagerly())
		self.inputlayer = GINN_inputLayer(self.data.W,units = len(self.data.W),il_batch_input_shape= (self.data_d0,self.data_d1,self.data_d2))#len(self.data.W) should be as same as forthcoming self.K.
		self.K = self.inputlayer.K
		self.K2 = self.K*2 # Edges from Concept layer to next layer. See Fig 1 of Ito et al.(2020)
		self.secondlayer = layers.Dense(self.K2, activation="tanh",kernel_initializer='random_normal',use_bias = False,)
		self.outputlayer = layers.Dense(2, activation='softmax') 
		self.inputlayer.extract_vars(self)
		# super(GINN_model, self).build(input_shape) #Not certain if insntantiating the parenct class,i.e., keras.Model, is desirable
	
	def call(self, inputs): # inputs = v^{BOW}_j
		# print('is eager in GINN_model call',tf.executing_eagerly())
		vCS = self.inputlayer(inputs) 
		#adjust the size for next layer input.
		local_vCS = tf.expand_dims(vCS,axis=1)
		V3 = self.secondlayer(local_vCS)
		y = self.outputlayer(V3)
		self.y = y 
		self.inputlayer.set_y(self.y)
		y_prob_pred = tf.math.reduce_max(y, axis=2,keepdims= True)
		return y_prob_pred 

class InputData(object):
	def __init__(self,batch_size = None,isDropRemainder = True, data_segment = 'train'):
		self.batch_size = batch_size
		self.isRemainderTrue = isDropRemainder  
		self.read_pickles()
		self.test_input = self.preprocess_input(data_segment= 'test') # run test first in this line 
		self.train_input = self.preprocess_input(data_segment= 'training') # run after self.test_input has been created
	
	def read_pickles(self):
		#import wegiht 
		with open('data/test_W.pkl', 'rb') as fin:
			self.W = pickle.load(fin)
		#import training_data(i.e. vbow) 
		with open('data/0_input_data.pkl', 'rb') as fin:
			self.training_data = pickle.load(fin)
		#import test_data
		with open('data/1_input_data.pkl', 'rb') as fin:
			self.test_data = pickle.load(fin)
		#import label for test data (positive = 1, negative = 0) 
		with open('data/0_labels.pkl', 'rb') as fin:
			self.labels_train = pickle.load(fin)
		#import label for test data (positive = 1, negative = 0) 
		with open('data/1_labels.pkl', 'rb') as fin:
			self.labels_test = pickle.load(fin)


		print('===End Reading Pickles==')
	
	def preprocess_input(self,data_segment = 'training'):
		"""
		change numpy ndarray training_data AND labels to tf.data.Dataset data.
		"""
		if data_segment == 'training':
			data = self.training_data
			labels = self.labels_train
		elif data_segment =='test':
			data = self.test_data
			labels = self.labels_test
		
		#1.Preprocess training data
		preprocessed_training_data = [x for x in data ]
		data_intermediatelist = []
		for cnt in range(len(preprocessed_training_data)):
			di = preprocessed_training_data[cnt][:].tolist()
			data_intermediatelist.append(sum(di,[])) # flatten
		self.data_frequencies_All_data = data_intermediatelist #Store data for invocation in input lyaer.
		data_frequencies = tf.expand_dims(tf.constant(data_intermediatelist,dtype = 'float32'),axis= 1) #expand dim at axis 1 to enable future propergation among layers.
		self.x = data_frequencies
		# self.x = tf.expand_dims(self.x,axis =2) #Adding dummy dimension for future preprocessing spefically fro Binary cross entropy and its argument 'reduction=tf.keras.losses.Reduction.NONE'.

		#2.Preprocess label
		self.formatted_labels=tf.expand_dims(tf.constant(labels[:].tolist(),dtype= 'float32'),axis=1)
		
		#3.Enter xs and ys to tf.data.Dataset
		self.row_dimension = len(self.x[0])
		self.feature_dimension = len(self.x[0][0])
		if self.isRemainderTrue == False:
			self.lastrow_num_of_data = len(self.x)  #-1 means index offset
		elif self.isRemainderTrue == True:
			self.lastrow_num_of_data = (len(self.x) // self.batch_size) * self.batch_size  #-1 means index offset
		#TODO take num of cluster and dimentions of data and add them as attributes of this call to use them 		

		inputs = tf.data.Dataset.from_tensor_slices((self.x, self.formatted_labels)).batch(self.batch_size,drop_remainder= self.isRemainderTrue)
		# print('top.input=', self.inputs)

		return inputs
		

#ToDo write decorator to save output to .txt file
def output_to_txt_file(f):
	def log():
		# os.system('script test.txt')
		print('start now')
		f()
	return log 

@output_to_txt_file
def main():
	data = InputData(batch_size=1, isDropRemainder = True, data_segment= 'train') 
	g_model = GINN_model(data)
	g_model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(),run_eagerly = True, metrics =['accuracy']) # you need 'run_eagerly = True' arg to run the whole process in eager mode.
	print(g_model.run_eagerly)
	g_model.fit(data.train_input, epochs = 3)
	g_model.summary()
	g_model.evaluate(data.test_input)

	#STUB
	metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
	y_true = np.array([[1, 1, 1],
                   	[1, 0, 0],
                   	[1, 1, 0]], np.int32)
	y_pred = np.array([[0.2, 0.6, 0.7],
                   	[0.2, 0.6, 0.6],
                   	[0.6, 0.8, 0.0]], np.float32)
	metric.update_state(y_true, y_pred)
	result = metric.result()
	print(result.numpy())

	# print(g_model.inputlayer.weights)

main()

