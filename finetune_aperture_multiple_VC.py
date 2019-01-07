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
import sys

occlusion_level=argv[1]
save_name=argv[2]
mod=argv[3]
VGG_MEAN = [123.68, 116.78, 103.94]
num_workers=4
batch_size=128
weight_decay=0.001
dropout_keep_prob=0.5
learning_rate1=0.005
learning_rate_decay_factor=0.1
num_epochs=20
finetunedir=os.path.join('/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint')
model_path='/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint/fine_tuned_with_portrait'
# scale_size = vgg.vgg_16.default_image_size

# # Runtime params
# checkpoints_dir = os.path.join('/home/xuyangf/project/ML_deliverables/Siamese_iclr17_tf-master/cache','checkpoints')
# tf.logging.set_verbosity(tf.logging.INFO)

# # Create the model, use the default arg scope to configure the batch norm parameters.
# with tf.device('/cpu:0'):
# 	input_images = tf.placeholder(tf.float32, [ batch_size,  scale_size,  scale_size, 3])

# with tf.variable_scope('vgg_16', reuse=False):
# 	with slim.arg_scope(vgg.vgg_arg_scope()):
# 		_, vgg_end_points = vgg.vgg_16( input_images, is_training=True)

train_dir='/data2/xuyangf/OcclusionProject/NaiveVersion/CroppedImage'
# val_dir='/data2/xuyangf/OcclusionProject/NaiveVersion/ValSet'
val_file_name='/data2/haow3/data/imagenet/dataset/val_crop_'+occlusion_level+'.txt'
train_filenames, train_labels = list_images(train_dir)
val_filenames, val_labels = list_images_from_txt(val_file_name)

train_dir2='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/top10vc'
train_filenames2, train_labels2 = list_images(train_dir2)
print(len(set(train_labels2)))


train_dir3='/data2/xuyangf/OcclusionProject/NaiveVersion/PortraitImages/train'
train_filenames3, train_labels3 = list_images_from_additional_set(train_dir3)

train_dir4='/data2/xuyangf/OcclusionProject/NaiveVersion/ApertureImage/train/bubble_image/top10vc_Portrait'
train_filenames4, train_labels4 = list_images_from_additional_set(train_dir4)
print(len(set(train_labels4)))


if mod == '1':
	train_filenames=train_filenames+train_filenames2+train_filenames3+train_filenames4
	train_labels=train_labels+train_labels2+train_labels3+train_labels4
else:
	train_filenames=train_filenames+train_filenames2+train_filenames3
	train_labels=train_labels+train_labels2+train_labels3	

# train_filenames=train_filenames+train_filenames2
# train_labels=train_labels+train_labels2

num_classes = len(set(train_labels))
print(num_classes)

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

        # Preprocessing (for training)
        # (3) Take a random 224x224 crop to the scaled image
        # (4) Horizontally flip the image with probability 1/2
        # (5) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
	def training_preprocess(image, label):
		crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
		flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

		means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
		centered_image = flip_image - means                                     # (5)

		return centered_image, label

        # Preprocessing (for validation)
        # (3) Take a central 224x224 crop to the scaled image
        # (4) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
	def val_preprocess(image, label):
		crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

		means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
		centered_image = crop_image - means                                     # (4)
		return centered_image, label

	train_filenames = tf.constant(train_filenames)
	train_labels = tf.constant(train_labels)
	train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
	train_dataset = train_dataset.map(_parse_function,
	    num_threads=num_workers, output_buffer_size=batch_size)
	train_dataset = train_dataset.map(training_preprocess,
	    num_threads=num_workers, output_buffer_size=batch_size)
	train_dataset = train_dataset.shuffle(buffer_size=50000)  # don't forget to shuffle
	batched_train_dataset = train_dataset.batch(batch_size)

	val_filenames = tf.constant(val_filenames)
	val_labels = tf.constant(val_labels)
	val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
	val_dataset = val_dataset.map(_parse_function,
	num_threads=num_workers, output_buffer_size=batch_size)
	val_dataset = val_dataset.map(val_preprocess,
	num_threads=num_workers, output_buffer_size=batch_size)
	batched_val_dataset = val_dataset.batch(batch_size)

	iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
	                                               batched_train_dataset.output_shapes)
	images, labels = iterator.get_next()

	train_init_op = iterator.make_initializer(batched_train_dataset)
	val_init_op = iterator.make_initializer(batched_val_dataset)

	# Indicates whether we are in training or in test mode
	is_training = tf.placeholder(tf.bool)

	vgg = tf.contrib.slim.nets.vgg
	with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
		logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
	                           dropout_keep_prob=dropout_keep_prob)

	variables_to_restore = tf.contrib.framework.get_variables_to_restore()
	init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

	# Initialization operation from scratch for the new "fc8" layers
	# `get_variables` will only return the variables whose name starts with the given pattern
	fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
	fc7_variables = tf.contrib.framework.get_variables('vgg_16/fc7')
	all_variables=fc8_variables+fc7_variables+tf.contrib.framework.get_variables('vgg_16/fc6')+tf.contrib.framework.get_variables('vgg_16/pool5')+tf.contrib.framework.get_variables('vgg_16/conv5')+tf.contrib.framework.get_variables('vgg_16/pool4')+tf.contrib.framework.get_variables('vgg_16/conv4')

	# fc8_init = tf.variables_initializer(fc8_variables)
	# ---------------------------------------------------------------------
	# Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
	# We can then call the total loss easily
	tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	loss = tf.losses.get_total_loss()

	# First we want to train only the reinitialized last layer fc8 for a few epochs.
	# We run minimize the loss only with respect to the fc8 variables (weight and bias).
	global_step = tf.Variable(0,trainable=False)
	learning_rate=tf.train.exponential_decay( learning_rate1,
                                      global_step,
                                      5000,
                                      learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
	fc8_optimizer = tf.train.MomentumOptimizer( learning_rate, 0.9)
	fc8_train_op = fc8_optimizer.minimize(loss, var_list=all_variables,global_step=global_step)

	# Then we want to finetune the entire model for a few epochs.
	# We run minimize the loss only with respect to all the variables.
	# tobeinitialized1 = tf.contrib.framework.get_variables('vgg_16/fc8')
	# tobeinitialized2=tf.contrib.framework.get_variables('vgg_16/fc7')
	# tobeinitialized=tobeinitialized1+tobeinitialized2
	# tobeinitialized.append(global_step)

	#print(tobeinitialized)
	#fc8_init = tf.variables_initializer(tobeinitialized)
	# Evaluation metrics
	init = tf.global_variables_initializer()
	prediction = tf.to_int32(tf.argmax(logits, 1))
	correct_prediction = tf.equal(prediction, labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()
	tf.get_default_graph().finalize()
	
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(graph=graph,config=config) as sess:
	sess.run(init) 
	init_fn(sess)  # load the pretrained weights
	#sess.run(fc8_init)  # initialize the new fc8 layer

	# Update only the last layer for a few epochs.
	for epoch in range(num_epochs):
	    # Run an epoch over the training data.
	    print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
	    # Here we initialize the iterator with the training set.
	    # This means that we can go through an entire epoch until the iterator becomes empty.
	    sess.run(train_init_op)
	    while True:
	        try:
	            _ = sess.run(fc8_train_op, {is_training: True})
	        except tf.errors.OutOfRangeError:
	            break
	    # Check accuracy on the train and val sets every epoch.
	    train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
	    val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
	    print('Train accuracy: %f' % train_acc)
	    print('Val accuracy: %f\n' % val_acc)
	    print(sess.run(global_step))
	    print(time.ctime(time.time()))
	    #print(sess.run(tf.contrib.framework.get_variables('vgg_16/fc8/biases/Momentum')))
	    #break
	saver.save(sess, os.path.join(finetunedir, save_name))
