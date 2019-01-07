from utils import *
from tensorflow.python.client import timeline
from datetime import datetime
# import network as vgg
from global_variables import *
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import random
import json
import cv2,os,glob,pickle,sys
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from GetDataPath import *
from ProjectUtils import *
import time

class TestVgg:
	def __init__(self,mylayer,network_name):
		vgg = tf.contrib.slim.nets.vgg
		self.VGG_MEAN = [123.68, 116.78, 103.94]
		self.batch_size = 1
		self.scale_size = vgg.vgg_16.default_image_size
		self.feature_size=224/pow(2,mylayer)
		self.featDim_set = [64, 128, 256, 512, 512] 
		self.mystring=tf.placeholder(tf.string)
		self.featuredim=self.featDim_set[int(mylayer)-1]
		tf.logging.set_verbosity(tf.logging.INFO)
		with tf.device('/cpu:0'):
			self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])
			self.input_features = tf.placeholder(tf.float32, [self.batch_size, self.feature_size, self.feature_size, self.featuredim])

		####################################################
		def _parse_function(filename):
			image_string = tf.read_file(filename)
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
			image = tf.cast(image_decoded, tf.float32)

			smallest_side = 256.0
			height, width = tf.shape(image)[0], tf.shape(image)[1]
			height = tf.to_float(height)
			width = tf.to_float(width)

			scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
			new_height = tf.to_int32(height * scale)
			new_width = tf.to_int32(width * scale)

			resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
			return resized_image

		def val_preprocess(image):
			crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

			means = tf.reshape(tf.constant(self.VGG_MEAN), [1, 1, 3])
			centered_image = crop_image - means                                     # (4)
			return centered_image

		####################################################

		self.myimage=val_preprocess(_parse_function(self.mystring))

		# with tf.variable_scope('vgg_16', reuse=False):
		with slim.arg_scope(vgg.vgg_arg_scope()):
			self.final_result, _ = vgg.vgg_16(self.input_images,num_classes=100, is_training=False,dropout_keep_prob=1)
			# self.final_result_part1 = vgg.vgg_16_part1(self.input_images,num_classes=100, is_training=False,dropout_keep_prob=1)
			# self.final_result_part2 =vgg.vgg_16_part2(self.input_features,num_classes=100, is_training=False,dropout_keep_prob=1)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

		# model_path=os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint',network_name)
		# variables_to_restore = tf.contrib.framework.get_variables_to_restore()
		# init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
		# init_fn(self.sess)

		restorer = get_init_restorer()
		init_op = tf.global_variables_initializer()
		checkpoints_dir = os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint')
		restorer.restore(self.sess, os.path.join(checkpoints_dir, network_name))
		print('load finish')


	def getacc(self,thestring,category):
		batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
		feed_dict = {self.mystring: thestring}
		theimage = self.sess.run(self.myimage, feed_dict=feed_dict)
		batch_images[0]=theimage
		feed_dict = {self.input_images: batch_images}
		out_features=self.sess.run(self.final_result, feed_dict=feed_dict)
		base=out_features[0][category]
		prob=1/(1+np.exp(-base))
		recognizable=True
		prediction=category
		for i in range(0,100):
			if out_features[0][i]>base:
				recognizable=False
				prediction=i
				break
		return recognizable, out_features[0], prediction,prob
