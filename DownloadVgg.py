import tensorflow as tf
from tensorflow.python.client import timeline
from global_variables import *
from slim.nets import vgg
from slim.datasets import dataset_utils

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
checkpoints_dir = os.path.join(g_cache_folder, 'checkpoints')
if not tf.gfile.Exists(checkpoints_dir):
   tf.gfile.MakeDirs(checkpoints_dir)
dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
print("yes")
