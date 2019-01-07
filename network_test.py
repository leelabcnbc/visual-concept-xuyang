from utils import *
from tensorflow.python.client import timeline
from datetime import datetime
import network as vgg
from global_variables import *
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import random
import json
from scipy.optimize import linear_sum_assignment
from ProjectUtils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from sys import argv
import time

val_directory_name=argv[1]
mymodel=argv[2]
VGG_MEAN = [123.68, 116.78, 103.94]
num_workers=4
batch_size=128
weight_decay=0.001
dropout_keep_prob=0.5
learning_rate1=0.0005
learning_rate_decay_factor=0.1

model_path=os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint',mymodel)


# val_filenames, val_labels = list_images_from_additional_set(val_directory_name)
# val_filenames, val_labels = list_images_from_txt(val_directory_name)
val_filenames, val_labels = list_images(val_directory_name)


num_classes = 100
print('num_classes: '+str(num_classes))
graph = tf.Graph()
with graph.as_default():
	def _parse_function(filename, label):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
		image = tf.cast(image_decoded, tf.float32)

		smallest_side = 224.0
		height, width = tf.shape(image)[0], tf.shape(image)[1]
		height = tf.to_float(height)
		width = tf.to_float(width)

		scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
		new_height = tf.to_int32(height * scale)
		new_width = tf.to_int32(width * scale)

		resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
		return resized_image, label

	def val_preprocess(image, label):
		crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

		means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
		centered_image = crop_image - means                                     # (4)
		return centered_image, label

	val_filenames = tf.constant(val_filenames)
	val_labels = tf.constant(val_labels)
	val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
	val_dataset = val_dataset.map(_parse_function,
	num_threads=num_workers, output_buffer_size=batch_size)
	val_dataset = val_dataset.map(val_preprocess,
	num_threads=num_workers, output_buffer_size=batch_size)
	batched_val_dataset = val_dataset.batch(batch_size)

	iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types,
	                                               batched_val_dataset.output_shapes)
	images, labels = iterator.get_next()

	val_init_op = iterator.make_initializer(batched_val_dataset)

	# Indicates whether we are in training or in test mode
	is_training = tf.placeholder(tf.bool)

	vgg = tf.contrib.slim.nets.vgg
	with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
		logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
	                           dropout_keep_prob=dropout_keep_prob)

	# Specify where the model checkpoint is (pretrained weights).

	# Restore only the layers up to fc7 (included)
	# Calling function `init_fn(sess)` will load all the pretrained weights.
	variables_to_restore = tf.contrib.framework.get_variables_to_restore()
	init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

	init = tf.global_variables_initializer()
	prediction = tf.to_int32(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()
	tf.get_default_graph().finalize()
	
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print(time.ctime(time.time()))
with tf.Session(graph=graph,config=config) as sess:
	sess.run(init) 
	init_fn(sess)  # load the pretrained weights

	val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
	print('original test')
	print('Val accuracy: %f\n' % val_acc)
	print(time.ctime(time.time()))