from utils import *
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
#from global_variables import *
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
	def __init__(self,mylayer):
		self.batch_size = 1
		self.scale_size = vgg.vgg_16.default_image_size
		self.feature_size=224/pow(2,mylayer)
		self.featDim_set = [64, 128, 256, 512, 512] 

		self.featuredim=self.featDim_set[int(mylayer)-1]
		tf.logging.set_verbosity(tf.logging.INFO)

		with tf.device('/cpu:0'):
			self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])
			self.input_features = tf.placeholder(tf.float32, [self.batch_size, self.feature_size, self.feature_size, self.featuredim])
		with tf.variable_scope('vgg_16', reuse=False):
			with slim.arg_scope(vgg.vgg_arg_scope()):
				self.final_result, _ = vgg.vgg_16(self.input_images,num_classes=100, is_training=False,dropout_keep_prob=1)
				self.final_result_part1 = vgg.vgg_16_part1(self.input_images,num_classes=100, is_training=False,dropout_keep_prob=1)
				self.final_result_part2 =vgg.vgg_16_part2(self.input_features,num_classes=100, is_training=False,dropout_keep_prob=1)

		restorer = get_init_restorer()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		#init_op = tf.global_variables_initializer()
		self.sess = tf.Session(config=config)

		checkpoints_dir = os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint')
		restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned'))
		print('load finish')

	def getprob(self,image,category):
		batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
		batch_images[0],axx,axxx = process_image(image, '/home/xuyangf', augment=0)
		#batch_images[0]=cv2.resize(image,(224,224))-np.float32([[[104., 117., 124.]]])
		feed_dict = {self.input_images: batch_images}
		out_features = self.sess.run(self.final_result, feed_dict=feed_dict)
		return out_features[0][category]

	def img2feature(self,image):
		batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
		batch_images[0],axx,axxx = process_image(image, '/home/xuyangf', augment=0)
		#batch_images[0]=cv2.resize(image,(224,224))-np.float32([[[104., 117., 124.]]])
		feed_dict = {self.input_images: batch_images}
		out_features = self.sess.run(self.final_result_part1, feed_dict=feed_dict)
		return out_features

	def feature2result(self,feature,category):
		print(feature.shape)
		batch_features = np.ndarray([self.batch_size, self.feature_size, self.feature_size, self.featuredim])
		batch_features=feature
		feed_dict = {self.input_features: batch_features}
		out_features = self.sess.run(self.final_result_part2, feed_dict=feed_dict)
		return out_features[0][category]

	def get_image_info(self,image):
		batch_images = np.ndarray([1, self.scale_size, self.scale_size, 3])
		batch_images[0],axx,axxx = process_image(image, '/home/xuyangf', augment=0)
		feed_dict = {self.input_images: batch_images}
		out_features = self.sess.run(self.final_result, feed_dict=feed_dict)
		print(decode_predictions(out_features[0:1,:])[0:2])

# image_path=LoadImage('0')
# paths = image_path[0:300]

# feature_list = []

# for i in range(-(-len(paths) // self.batch_size)):
#     batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
#     for j in range(self.batch_size):
#         # read paths
#         if i * self.batch_size + j >= len(paths):
#             break
#         img = cv2.imread(paths[i * self.batch_size + j], cv2.IMREAD_UNCHANGED)
#         #print(paths[i * self.batch_size + j])
#         #print(img.shape)
#         batch_images[j],axx,axxx = process_image(img, paths[i * self.batch_size + j], augment=0)
#         #batch_images[j]=cv2.resize(img,(224,224))-np.float32([[[104., 117., 124.]]])
#     feed_dict = {input_images: batch_images}
#     out_features = self.sess.run(final_result, feed_dict=feed_dict)
#     feature_list.append(out_features)

# features = np.concatenate(feature_list)

# print('resultget finish')
# print(features.shape)

# sumtop1=0
# sumtop5=0
# sumscore=0
# for i in range(0,300):
# 	if decode_predictions(features[0:300,:])[i][0][0]=='n02002556':
# 		sumtop1+=1
# 	for j in range(0,5):
# 		if decode_predictions(features[0:300,:])[i][j][0]=='n02002556':
# 			sumtop5+=1
# 			sumscore+=decode_predictions(features[0:300,:])[i][j][2]
# _e = time.time()

# print('sumtop1   '+str(sumtop1))
# print('sumtop5   '+str(sumtop5))
# print('sumscore   '+str(sumscore))

# print(decode_predictions(features[0:300,:])[0:2])
# print('running time seconds: {0}'.format((_e-_s)))
# print('finish')
