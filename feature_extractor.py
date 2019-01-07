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


class FeatureExtractor:
    def __init__(self, which_layer='pool4', which_snapshot=200000, from_scratch=False):
        # params
        self.batch_size = 1
        self.scale_size = vgg.vgg_16.default_image_size

        # Runtime params
        checkpoints_dir = '/data2/xuyangf/OcclusionProject/NaiveVersion/checkpoint'
        tf.logging.set_verbosity(tf.logging.INFO)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with tf.device('/cpu:0'):
            self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.scale_size, self.scale_size, 3])

        with tf.variable_scope('vgg_16', reuse=False):
            with slim.arg_scope(vgg.vgg_arg_scope()):
                _, vgg_end_points =  vgg.vgg_16(self.input_images,num_classes=100, is_training=False,dropout_keep_prob=1)
        # self.pool4 = vgg_end_points['vgg_16/pool4']
        # with tf.variable_scope('VC', reuse=False):
        #     self.tight_loss, self.tight_end_points = online_clustering(self.pool4, 512)
        if which_layer[0]>'0' and which_layer[0]<='9':
            self.features = vgg_end_points['vgg_16_'+which_layer[0]+'/' + which_layer[1:]]
        else:
            self.features = vgg_end_points['vgg_16/' + which_layer]  # TODO

        # Create restorer and saver
        restorer = get_init_restorer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.global_variables_initializer()
        # Run the session:
        self.sess = tf.Session(config=config)
        print(str(datetime.now()) + ': Start Init')
        if which_snapshot == 0:  # Start from a pre-trained vgg ckpt
            if from_scratch:
                self.sess.run(init_op)
            else:
                restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned'))
        else:  # Start from the last time
            # sess.run(init_op)
            restorer.restore(self.sess, os.path.join(checkpoints_dir, 'fine_tuned-' + str(which_snapshot)))
        print(str(datetime.now()) + ': Finish Init')

        # visualize first layer conv filters
        # conv1_1 = restorer._var_list[0]
        # conv1_1_weights = self.sess.run(conv1_1) * 0.5 + 0.5
        # fig = plt.figure(figsize=(16, 9), dpi=300)
        # for i in range(64):
        #     ax = fig.add_subplot(8, 8, i + 1)
        #     ax.imshow(conv1_1_weights[:, :, :, i])
        #     ax.get_xaxis().set_ticks([])
        #     ax.get_xaxis().set_ticklabels([])
        #     ax.get_yaxis().set_ticks([])
        #     ax.get_yaxis().set_ticklabels([])
        # fig.savefig(os.path.join(g_cache_folder, 'weights.eps'))
        # fig.clear()

    def extract_from_paths(self, paths):
        feature_list = []
        image_list = []
        blank_list=[]
        for i in range(-(-len(paths) // self.batch_size)):
            batch_images = np.ndarray([self.batch_size, self.scale_size, self.scale_size, 3])
            batch_blank=np.ndarray([self.batch_size,2])
            for j in range(self.batch_size):
                # read paths
                if i * self.batch_size + j >= len(paths):
                    break
                img = cv2.imread(paths[i * self.batch_size + j], cv2.IMREAD_UNCHANGED)
                #print(paths[i * self.batch_size + j])
                #print(img.shape)
                batch_images[j],batch_blank[j][0],batch_blank[j][1] = process_image(img, paths[i * self.batch_size + j], augment=0)
                # batch_images[j],batch_blank[j][0],batch_blank[j][1] = process_image2(img)
            out_features = self.extract_from_batch_images(batch_images)
            feature_list.append(out_features)
            image_list.append(batch_images)
            blank_list.append(batch_blank)

        features = np.concatenate(feature_list)
        images = np.concatenate(image_list)
        blanks= np.concatenate(blank_list)
        return features[:len(paths), :], images[:len(paths), :], blanks[:len(paths), :]

    def extract_from_batch_images(self, batch_images):
        feed_dict = {self.input_images: batch_images}
        # [out_features, out_end_points, out_tight_loss] = self.sess.run([self.features, self.tight_end_points, self.tight_loss], feed_dict=feed_dict)
        out_features = self.sess.run(self.features, feed_dict=feed_dict)
        return out_features

