import sys
sys.path.append('..')
import tensorflow as tf
from tensorflow.layers import *
import numpy as np
from net_utils import *




class ResNet():
	
	def __init__(self, img, image_shape=[224,224], classes_num=19, training=True, regularizer=None):
		self.img = img
		self.image_shape = image_shape
		self.classes_num = classes_num
		self.regularizer = regularizer
		self.is_training = training
		
	def build_net(self, mode='lkp'):
		''' The backbone based on ResNet50 '''
		img = self.img
		with tf.variable_scope("Res_encode"):
			with tf.variable_scope("Res_initlayer"):
				
				conv1 = self.conv_bn(img, 7, 64, 2, "conv1")
				# pool1 = max_pooling2d(conv1, 2, 2, padding='same',name='pool1')
				conv2 = self.conv_bn(conv1, 5, 64, 2, "conv2")
				bn_p =  batch_normalization(conv2, epsilon=1e-12, name='bn2', training=self.is_training)
				relu_p = tf.nn.relu(bn_p,name='relu')
			#1/4
			with tf.variable_scope("ResBlock1"):
				
				self.block1_1 = self.bottleneck_res(relu_p, [64, 64, 256], "block1_1", True)
				self.block1_2 = self.bottleneck_res(self.block1_1, [64, 64, 256], "block1_2")
				self.block1_3 = self.bottleneck_res(self.block1_2, [64, 64, 256], "block1_3")
			
			#1/8
			with tf.variable_scope("ResBlock2"):
				
				self.block2_1 = self.bottleneck_res(self.block1_3, [128, 128, 512], "block2_1", True, 2)
				self.block2_2 = self.bottleneck_res(self.block2_1, [128, 128, 512], "block2_2")
				self.block2_3 = self.bottleneck_res(self.block2_2, [128, 128, 512], "block2_3")
				self.block2_4 = self.bottleneck_res(self.block2_3, [128, 128, 512], "block2_4")
				
			#1/16
			with tf.variable_scope("ResBlock3"):
				
				self.block3_1 = self.bottleneck_res(self.block2_4, [256, 256, 1024], "block3_1", True, 2)
				self.block3_2 = self.bottleneck_res(self.block3_1, [256, 256, 1024], "block3_2")
				self.block3_3 = self.bottleneck_res(self.block3_2, [256, 256, 1024], "block3_3")
				self.block3_4 = self.bottleneck_res(self.block3_3, [256, 256, 1024], "block3_4")
				self.block3_5 = self.bottleneck_res(self.block3_4, [256, 256, 1024], "block3_5")
				self.block3_6 = self.bottleneck_res(self.block3_5, [256, 256, 1024], "block3_6")
			
			#1/32
			with tf.variable_scope("ResBlock4"):
				
				self.block4_1 = self.bottleneck_res(self.block3_6, [512, 512, 2048], "block4_1", True, 2)
				self.block4_2 = self.bottleneck_res(self.block4_1, [512, 512, 2048], "block4_2")
				self.block4_3 = self.bottleneck_res(self.block4_2, [512, 512, 2048], "block4_3")
		with tf.variable_scope('Decoder'):
			shape = list(map(lambda x :x//16, self.image_shape))
			baseline_out = Upsample(self.block4_3, shape)
			x = tf.concat([baseline_out, self.seperable_conv_bn(self.block3_6, 1, 512, 1, 'block3_feature_extractor')], axis=-1)
			x = self.seperable_conv_bn(x, 1, 512, 1, 'U_fuse1')
			x = tf.nn.relu(x)
			
			
			baseline_out = Upsample(x, list(map(lambda x :x//8, self.image_shape)))
			x = tf.concat([baseline_out, self.seperable_conv_bn(self.block2_4, 1, 256, 1, 'block2_feature_extractor')], axis=-1)
			x = self.seperable_conv_bn(x, 1, 256, 1, 'U_fuse2')
			x = tf.nn.relu(x)
			
			baseline_out = Upsample(x, list(map(lambda x :x//4, self.image_shape)))
			x = tf.concat([baseline_out, self.seperable_conv_bn(self.block1_3, 1, 128, 1, 'block1_feature_extractor')], axis=-1)
			x = self.seperable_conv_bn(x, 1, 128, 1, 'U_fuse3')
			x = tf.nn.relu(x)
			
			if mode == 'lkp':
				sppout = tf.nn.relu(LKP(x, 128,
						self.regularizer, self.is_training)([1,3,5,7], 'LargeKernelPooling'))
				print('lkp')
			elif mode == 'aspp':
				sppout = tf.nn.relu(ASPP(x, 128, 
						self.regularizer, self.is_training)('ASPP'))
				print('aspp')
			elif mode == 'laspp':
				sppout = tf.nn.relu(LASPP(x, 128, 
						self.regularizer, self.is_training)('LASPP'))
			elif mode == 'denseaspp':
				sppout = tf.nn.relu(build_dense_aspp(x))
				
			print(sppout)
			x = conv2d(sppout, 64, 3, padding='same',
				kernel_initializer=tf.variance_scaling_initializer(),
				activation=tf.nn.relu)
			
			x = Upsample(x, self.image_shape)
		with tf.variable_scope('logits'):
			x = conv2d(x, self.classes_num, 1,
				kernel_initializer=tf.variance_scaling_initializer(),
				activation=None)#Set activation as tf.nn.softmax() when studying the proposed losses
		return x
		
	def batch_norm(self, x):
		"""
		Batchnorm
		"""
		_BATCH_NORM_DECAY = 0.99
		_BATCH_NORM_EPSILON = 1e-12
		return batch_normalization(inputs=x, axis = -1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training)
	
	def seperable_conv_bn(self, x, size, filters, stride, name):
		x = separable_conv2d(x, filters, size, strides=stride, padding='same',
								use_bias=True,
								depthwise_initializer=tf.variance_scaling_initializer(),
								pointwise_initializer=tf.variance_scaling_initializer(),
								depthwise_regularizer=self.regularizer,
								pointwise_regularizer=self.regularizer,
								bias_regularizer=self.regularizer,
								name=name,)
		x = self.batch_norm(x)
		x = tf.nn.relu(x)
		return x
	
	def conv_bn(self, x, size, filters, stride, name, dilation_rate=1):
		"""
		conv_layer+bn+relu
		"""
		x = conv2d(x, filters, size, strides=stride, padding='same', 
			dilation_rate=dilation_rate,
			kernel_initializer=tf.variance_scaling_initializer(),
			kernel_regularizer=self.regularizer,
			name=name)
		x = self.batch_norm(x)
		x = tf.nn.relu(x)
		return x
	
	def bottleneck_res(self, x, channel_list, name, change_dimension=False, block_stride=1, dialation=1, regularizer=None):
		"""
		bottlneeck_block in ResNet
		"""
		if (change_dimension):
			short_cut_conv = self.seperable_conv_bn(x, 3, channel_list[2], block_stride, name + "_ShortcutConv")
			block_conv_input = self.batch_norm(short_cut_conv)
		else:
			block_conv_input = x
	
		x = self.conv_bn(block_conv_input, 1, channel_list[0], 1, name+'conv1x1_1')
		x = self.conv_bn(x, 3, channel_list[1], 1, name+'conv3x3',)
		x = self.conv_bn(x, 1, channel_list[2], 1, name+'conv1x1_2',)
		x = tf.add(block_conv_input, x)
		x = tf.nn.relu(x)
		return x
	
	def deconv_layer(self, x):
		pass