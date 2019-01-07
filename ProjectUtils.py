import json
import warnings
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from GetDataPath import *

def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.
    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: in case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    fpath = '/data2/xuyangf/OcclusionProject/utils/my_class_index.json'
    CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)

    return results
def list_images_from_txt(fname):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    # labels = os.listdir(directory)
    # files_and_labels = []
    # for label in labels:
    #     for f in os.listdir(os.path.join(directory, label)):
    #         files_and_labels.append((os.path.join(directory, label, f), label))
    # print(files_and_labels)
    files_and_labels = []
    # for file in os.listdir(directory):
    #     files_and_labels.append((os.path.join(directory,file),file[:9]))
    directory="/data2/haow3/data/imagenet/dataset/"
    for line in open(fname):
        s=line.split()[0]
        files_and_labels.append((os.path.join(directory,s),s[s.find('/')+1:s.find('/')+10]))
    filenames,  labels = zip(*files_and_labels)
    filenames = list(filenames)

    # labels = list(labels)
    # unique_labels = list(set(labels))

    # label_to_int = {}
    fpath = '/data2/xuyangf/OcclusionProject/utils/my_class_index.json'
    CLASS_INDEX = json.load(open(fpath))

    unique_labels=[]
    for i in range(0,100):
        unique_labels.append(CLASS_INDEX[str(i)][0])
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]
    return filenames, labels

def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    # labels = os.listdir(directory)
    # files_and_labels = []
    # for label in labels:
    #     for f in os.listdir(os.path.join(directory, label)):
    #         files_and_labels.append((os.path.join(directory, label, f), label))
    # print(files_and_labels)
    files_and_labels = []
    for file in os.listdir(directory):
        files_and_labels.append((os.path.join(directory,file),file[:9]))
    filenames,  labels = zip(*files_and_labels)
    filenames = list(filenames)

    # labels = list(labels)
    # unique_labels = list(set(labels))

    # label_to_int = {}
    fpath = '/data2/xuyangf/OcclusionProject/utils/my_class_index.json'
    CLASS_INDEX = json.load(open(fpath))

    unique_labels=[]
    for i in range(0,100):
        unique_labels.append(CLASS_INDEX[str(i)][0])
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]
    return filenames, labels

def list_images_from_additional_set(directory):
    filenames=[]
    labels=[]
    for file in os.listdir(directory):
        filenames.append(os.path.join(directory,file))
        labels.append(100)
    return filenames,labels

def get_vc_result(sess, vc_result, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    all_vc_result=[]
    while True:
        try:
            batch_vc_result = sess.run(vc_result, {is_training: False})
            all_vc_result.append(batch_vc_result)
        except tf.errors.OutOfRangeError:
            break
    return     np.concatenate(all_vc_result,axis=0)

def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc

def check_prob(sess, logit, is_training, dataset_init_op):
    """
    Check the prob
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            all_logit = sess.run(logit, {is_training: False})
        except tf.errors.OutOfRangeError:
            break
    return all_logit
