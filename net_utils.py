import tensorflow as tf
from tensorflow.layers import *
import numpy as np
from tensorflow.contrib import slim
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.contrib import slim
class LKP():
	def __init__(self, data, filters, regularizer=None, training=True):
		self.data = data
		self.filters = filters
		self.regularizer = regularizer
		self.is_training = training
	
	def __call__(self, size_list, name):
		feature_map_size = tf.shape(self.data)
		
		# Global average pooling
		image_features = tf.reduce_mean(self.data, [1, 2], keep_dims=True)
		image_features = self.conv_bn(image_features, [1, 1], 1, name+'Global_pooling')
		image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
		
		branch1 = self.dense_lkp(self.data, size_list[0], name+'branch1')
		branch2 = self.dense_lkp(self.data, size_list[1], name+'branch2')
		branch3 = self.dense_lkp(self.data, size_list[2], name+'branch3')
		branch4 = self.dense_lkp(self.data, size_list[3], name+'branch4')
		x = tf.concat([branch1, branch2, branch3, branch4, image_features], axis=-1)
		x = self.conv_bn(x, 1, 1, name+'fuse')
		x = self.bn_layer(x)
		x = tf.nn.elu(x)
		return x
			
			
	def lkp_layer(self, x, size, name, dilation=1):
		x = self.bn_layer(x)
		x = tf.nn.elu(x)
		short_cut = x
		if size > 3:
			# x1 = self.conv_bn(x, [3,size], [1,dilation], name+'conv1')
			# # x = tf.add(short_cut, x)
			# x2 = self.conv_bn(x, [size,3], [dilation,1], name+'conv2')
			# x = tf.add(x1, x2)
			x = self.conv_bn(x, [3,size], [1,dilation], name+'conv1')
			x = self.conv_bn(x, [size,3], [dilation,1], name+'conv2')
			
			return x
			
		elif size ==3 :
			x = self.conv_bn(x, size, dilation, name+'conv1')
			# x = tf.add(short_cut, x)
			return x
			
		else:
			x = self.conv_bn(x, size, 1, name+'conv1')
			# x = tf.add(short_cut, x)
			return x
	
	def dense_lkp(self, x, size, name, layers=3):
		if size >2:
			x = self.lkp_layer(x, size, name+'_{}'.format(0), 1)
			for i in range(layers-1):
				x = self.lkp_layer(x, size, name+'_{}'.format(i+1), i+2)
			return x
		else:
			x = self.lkp_layer(x, size, name+'_{}'.format(0), 1)
			return x
		
	def conv_bn(self, x, size, dilation_rate, name):
		"""
		conv_layer+bn+elu
		"""
		x = conv2d(x, self.filters, size, strides=1, padding='same', 
			dilation_rate=dilation_rate,
			kernel_initializer=tf.variance_scaling_initializer(),
			kernel_regularizer=self.regularizer,
			name=name)
		x = self.bn_layer(x)
		x = tf.nn.elu(x)
		return x
		
	def bn_layer(self, x):
		"""
		Batchnorm
		"""
		_BATCH_NORM_DECAY = 0.99
		_BATCH_NORM_EPSILON = 1e-12
		return batch_normalization(inputs=x, axis = -1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training)
	
	def seperable_conv(self, x, size, dilation, name):
		return separable_conv2d(x, self.filters, size, padding='same',
								dilation_rate=dilation, depth_multiplier=1, use_bias=True,
								depthwise_initializer=tf.variance_scaling_initializer(),
								pointwise_initializer=tf.variance_scaling_initializer(),
								depthwise_regularizer=self.regularizer,
								pointwise_regularizer=self.regularizer,
								bias_regularizer=self.regularizer,
								name=name,)


class ASPP():
	def __init__(self, data, filters, regularizer=None, training=True):
		self.data = data
		self.filters = filters
		self.regularizer = regularizer
		self.is_training = training
	
	def __call__(self, name='ASPP', rate_list=[6, 12, 18]):
	
		feature_map_size = tf.shape(self.data)
		
		# Global average pooling
		image_features = tf.reduce_mean(self.data, [1, 2], keep_dims=True)
		
		image_features = slim.conv2d(image_features, self.filters, [1, 1], activation_fn=None)
		image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
		
		atrous_pool_block_1 = slim.conv2d(self.data, self.filters, [1, 1], activation_fn=None)
		
		
		atrous_pool_block_6 = slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[0], activation_fn=None)
		
		
		atrous_pool_block_12 = slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[1], activation_fn=None)
		
		
		atrous_pool_block_18 =  slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[2], activation_fn=None)
		
		
		net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)
		
		return slim.conv2d(net, self.filters, [1, 1], activation_fn=None)
		
		

		
		
class LASPP():
	def __init__(self, data, filters, regularizer=None, training=True):
		self.data = data
		self.filters = filters
		self.regularizer = regularizer
		self.is_training = training
	
	def __call__(self, name='LASPP', rate_list=[6, 12, 18]):
	
		feature_map_size = tf.shape(self.data)
		
		# Global average pooling
		image_features = tf.reduce_mean(self.data, [1, 2], keep_dims=True)
		
		image_features = slim.conv2d(image_features, self.filters, [1, 1], activation_fn=None)
		image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
		
		atrous_pool_block_1 = slim.conv2d(self.data, self.filters, [1, 1], activation_fn=None)
		# large_block_1 = self.large_block(self.data, 14, 1)
		# atrous_pool_block_1 = tf.add(atrous_pool_block_1, large_block_1)
		
		atrous_pool_block_6 = slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[0], activation_fn=None)
		large_block_6 = self.large_block(self.data, rate_list[0]-1, 1)
		atrous_pool_block_6 = tf.add(atrous_pool_block_6, large_block_6)
		
		atrous_pool_block_12 = slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[1], activation_fn=None)
		large_block_12 = self.large_block(self.data, rate_list[1]-1, 1)
		atrous_pool_block_12 = tf.add(atrous_pool_block_12, large_block_12)
		
		atrous_pool_block_18 =  slim.conv2d(self.data, self.filters, [3, 3], rate=rate_list[2], activation_fn=None)
		large_block_18 = self.large_block(self.data, rate_list[2]-1, 1)
		atrous_pool_block_18 = tf.add(atrous_pool_block_18, large_block_18)
		
		net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)
		
		return slim.conv2d(net, self.filters, [1, 1], activation_fn=None)
		
		
	def large_block(self, x, size, dilation_rate,):
		x1 = slim.separable_conv2d(x, self.filters, [1, size],
			depth_multiplier=3, rate=dilation_rate, activation_fn=None)
		x1 = slim.separable_conv2d(x, self.filters, [size, 1],
			depth_multiplier=3, rate=dilation_rate, activation_fn=None)
		
		x2 = slim.separable_conv2d(x, self.filters, [size, 1],
			depth_multiplier=3, rate=dilation_rate, activation_fn=None)
		x2 = slim.separable_conv2d(x, self.filters, [1, size],
			depth_multiplier=3, rate=dilation_rate, activation_fn=None)
		return tf.add(x1, x2)
		


def laplacian_edge(image):
	static_image_shape = image.get_shape()
	
	image_shape = array_ops.shape(image)
	kernels = np.ones([3,3,1,1])
	kernels[2,2,:,:] = -8
	
	# kernels = np.ones([9,9,1,1])
	# kernels[3:6,3:6,:,:] = -8
	
	kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

	kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
	name='laplacian_filters')

	# Use depth-wise convolution to calculate edge maps per channel.
	pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
	# pad_sizes = [[0, 0], [4, 4], [4, 4], [0, 0]]
	padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

	strides = [1, 1, 1, 1]
	output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
	# output = nn.conv2d(padded, kernels_tf, strides, 'VALID')
	output = array_ops.reshape(output, shape=static_image_shape)
	return output


#loss使用cos,norm,softmax,angle
def pam(x):
	B, H, W, C = x.get_shape().as_list()
	proj_query = tf.reshape(x, [B,H*W,C])
	proj_key = tf.transpose(x_reshape, perm=[0])
	return tf.matmul(proj_query, proj_key) 
	

def Upsample(x, shape):
	return tf.image.resize_bilinear(x, size=tf.cast(shape, tf.int32))

def DilatedConvBlock(inputs, n_filters, rate=1, kernel_size=[3, 3]):
	"""
	Basic dilated conv block 
	Apply successivly BatchNormalization, ReLU nonlinearity, dilated convolution 
	"""
	net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
	net = slim.conv2d(net, n_filters, kernel_size, rate=rate, activation_fn=None, normalizer_fn=None)
	return net



def build_dense_aspp(inputs, is_training=True):
	

	init_features = inputs
	### First block, rate = 3
	d_3_features = DilatedConvBlock(init_features, n_filters=256, kernel_size=[1, 1])
	d_3 = DilatedConvBlock(d_3_features, n_filters=64, rate=3, kernel_size=[3, 3])

	### Second block, rate = 6
	d_4 = tf.concat([init_features, d_3], axis=-1)
	d_4 = DilatedConvBlock(d_4, n_filters=256, kernel_size=[1, 1])
	d_4 = DilatedConvBlock(d_4, n_filters=64, rate=6, kernel_size=[3, 3])

	### Third block, rate = 12
	d_5 = tf.concat([init_features, d_3, d_4], axis=-1)
	d_5 = DilatedConvBlock(d_5, n_filters=256, kernel_size=[1, 1])
	d_5 = DilatedConvBlock(d_5, n_filters=64, rate=12, kernel_size=[3, 3])

	### Fourth block, rate = 18
	d_6 = tf.concat([init_features, d_3, d_4, d_5], axis=-1)
	d_6 = DilatedConvBlock(d_6, n_filters=256, kernel_size=[1, 1])
	d_6 = DilatedConvBlock(d_6, n_filters=64, rate=18, kernel_size=[3, 3])

	### Fifth block, rate = 24
	d_7 = tf.concat([init_features, d_3, d_4, d_5, d_6], axis=-1)
	d_7 = DilatedConvBlock(d_7, n_filters=256, kernel_size=[1, 1])
	d_7 = DilatedConvBlock(d_7, n_filters=64, rate=24, kernel_size=[3, 3])

	full_block = tf.concat([init_features, d_3, d_4, d_5, d_6, d_7], axis=-1)
	

	return net

def gridding(inputs, filters, is_training=True):
	net = slim.conv2d(inputs, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	net = slim.conv2d(net, filters, 3, rate=2, activation_fn=None, normalizer_fn=None)
	net = tf.nn.relu(slim.batch_norm(net, fused=True, is_training=is_training))
	return net