import tensorflow as tf
import numpy as np
import os
import glob
import math
import collections
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from utils import *

# Examples = collections.namedtuple("Examples", "inputs, count, steps_per_epoch")

BATCH_SIZE = 20
IMAGE_SIZE = 176
data_path = "./cityscapes/train"


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis = 2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def load_data(content):
    raw_input = tf.image.decode_jpeg(contents = content)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype = tf.float32)

    print("raw image shape: ", raw_input.shape)

    raw_input = tf.identity(raw_input)
    raw_input.set_shape([None, None, 3])
    width = tf.shape(raw_input)[1]
    raw_input = raw_input[:,:width//2, :] * 2 - 1

    print("raw input shape: ", raw_input.shape)

    raw_input = tf.image.resize_images(raw_input, [IMAGE_SIZE, IMAGE_SIZE], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = rgb_to_lab(raw_input)
    # l, ab = tf.split(value=lab,num_or_size_splits = [1, 2], axis = -1, name = 'split')
    l, ab = tf.split(lab, [1,2], -1)
#    L_chan, a_chan, b_chan = preprocess_lab(lab)
    return l, ab


def image_batch(img_path):
    file_names= []
    file_list = os.listdir(img_path)

    for i, files in enumerate(file_list):
        file_names.append(os.path.join(img_path, files))

    print(file_names)
    path_queue = tf.train.string_input_producer(file_names, shuffle = True)
    paths, contents = tf.WholeFileReader().read(path_queue)
    input_l, input_ab = load_data(contents)
    print(input_l)

    l_batch, ab_batch =\
        tf.train.batch([input_l, input_ab], batch_size= BATCH_SIZE)

    steps_per_epoch =int(math.ceil(len(file_names)/ BATCH_SIZE))
    return l_batch, ab_batch, len(file_names), steps_per_epoch

# with tf.Session() as sess:
#     tf.initialize_all_tables()
#     tf.initialize_all_variables()
#     l_value, count, _ = image_batch(data_path)
#     img = sess.run(l_value)
#
#     sess.close()
