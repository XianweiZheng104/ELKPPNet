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
