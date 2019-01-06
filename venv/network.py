import tensorflow as tf
import numpy as np
from operation import *

weight_decacy = 0.0
BATCH_SIZE = 40

# images: luminance
# ab_masks: ab of lab color range
def colorization_network(images, ab_masks):
    layers = []
    conv_filter = 64

    with tf.variable_scope("layer1"):
        # l, ab 따로 convolution
        image_conv1_1 = conv('image_conv1_1',images, [3, 3, 1, conv_filter], stride = 1, wd = weight_decacy)
        ab_conv1_1 = conv('ab_conv1_1', ab_masks, [3, 3, 2, conv_filter], stride = 1, wd = weight_decacy)

        # eltwise - summing two layers / 더한 후에 relu처리 한다.
        conv1_1 = tf.add(image_conv1_1, ab_conv1_1, name = 'conv1_1')
        relu1_1 = tf.nn.relu(conv1_1, name = 'relu1_1')

        conv1_2 = conv('conv1_2', relu1_1, [3, 3, conv_filter, conv_filter], stride = 1, wd = weight_decacy)
        relu1_2 = tf.nn.relu(conv1_2, name='relu1_2')

        bn_1 = batch_norm('conv1_2norm', relu1_2, train = True)
        # print("bn_1: ", bn_1)

        # convolution down sampling
        conv1 = conv_ds('conv1', bn_1, [1, 1, bn_1.get_shape()[-1], 1], stride = 2)
        layers.append(conv1)
        print("layer1: ",conv1)

    with tf.variable_scope("layer2"):
        conv2_1 = conv('conv2_1', layers[-1], [3, 3, conv_filter, conv_filter*2], stride = 1, wd =weight_decacy)
        relu2_1 = tf.nn.relu(conv2_1, name = 'relu2_1')

        conv2_2 = conv('conv2_2', relu2_1, [3, 3, conv_filter*2, conv_filter*2], stride = 1, wd = weight_decacy)
        relu2_2 = tf.nn.relu(conv2_2, name = 'relu2_2')

        bn_2 = batch_norm('conv2_2norm', relu2_2, train = True)

        conv2 = conv_ds('conv2', bn_2, [1, 1, bn_2.get_shape()[-1], 1], stride = 2)
        layers.append(conv2)
        print("layer2: ",conv2)

    with tf.variable_scope("layer3"):
        conv3_1 = conv('conv3_1', layers[-1], [3, 3, conv_filter*2, conv_filter*4], stride = 1, wd = weight_decacy)
        relu3_1 = tf.nn.relu(conv3_1, name = 'relu3_1')

        conv3_2 = conv('conv3_2', relu3_1, [3, 3, conv_filter*4, conv_filter*4],stride = 1, wd = weight_decacy)
        relu3_2 = tf.nn.relu(conv3_2, name = 'relu3_2')

        conv3_3 = conv('conv3_3', relu3_2, [3, 3, conv_filter*4, conv_filter*4], stride = 1, wd = weight_decacy)
        relu3_3 = tf.nn.relu(conv3_3, name = 'relu3_3')

        bn_3 = batch_norm('conv3_3norm', relu3_3, train = True)

        conv3 = conv_ds('conv3', bn_3, [1, 1, bn_3.get_shape()[-1], 1], stride = 2)
        layers.append(conv3)
        print("layer3: ",conv3)

    with tf.variable_scope("layer4"):
        conv4_1 = conv('conv4_1', layers[-1], [3, 3, conv_filter*4, conv_filter*8], stride = 1, wd = weight_decacy)
        relu4_1 = tf.nn.relu(conv4_1, name = 'relu4_1')

        conv4_2 = conv('conv4_2', relu4_1, [3, 3, conv_filter*8, conv_filter*8], stride = 1, wd = weight_decacy)
        relu4_2 = tf.nn.relu(conv4_2, name = 'relu4_2')

        conv4_3 = conv('conv4_3', relu4_2, [3, 3, conv_filter*8, conv_filter*8], stride = 1, wd = weight_decacy)
        relu4_3 = tf.nn.relu(conv4_3, name = 'relu4_3')

        bn_4 = batch_norm('conv4_3norm', relu4_3, train = True)
        layers.append(bn_4)
        print("layer4: ",bn_4)

    with tf.variable_scope("layer5"):
        conv5_1 = conv('conv5_1', layers[-1], [3, 3, conv_filter*8, conv_filter*8], stride = 1, dilation = 2, wd = weight_decacy)
        relu5_1 = tf.nn.relu(conv5_1, name = 'relu5_1')

        conv5_2 = conv('conv5_2', relu5_1, [3, 3, conv_filter*8,conv_filter*8], stride = 1, dilation = 2, wd = weight_decacy)
        relu5_2 = tf.nn.relu(conv5_2, name = 'relu5_2')

        conv5_3 = conv('conv5_3', relu5_2, [3, 3, conv_filter*8,conv_filter*8], stride = 1, dilation = 2, wd = weight_decacy)
        relu5_3 = tf.nn.relu(conv5_3, name = 'relu5_3')

        bn_5 = batch_norm('conv5_3norm', relu5_3, train = True)
        layers.append(bn_5)
        print("layer5: ",bn_5)

    with tf.variable_scope("layer6"):
        conv6_1 = conv('conv6_1', layers[-1], [3, 3, conv_filter*8, conv_filter*8], stride = 1, dilation = 2, wd = weight_decacy)
        relu6_1 = tf.nn.relu(conv6_1, name = 'relu6_1')

        conv6_2 = conv('conv6_2', relu6_1, [3, 3, conv_filter * 8, conv_filter * 8], stride=1, dilation=2, wd=weight_decacy)
        relu6_2 = tf.nn.relu(conv6_2, name='relu6_2')

        conv6_3 = conv('conv6_3', relu6_2, [3, 3, conv_filter * 8, conv_filter * 8], stride=1, dilation=2, wd=weight_decacy)
        relu6_3 = tf.nn.relu(conv6_3, name='relu6_3')

        bn_6 = batch_norm('conv6_3norm', relu6_3, train = True)
        layers.append(bn_6)
        print("layer6: ",bn_6)

    with tf.variable_scope("layer7"):
        conv7_1 = conv('conv7_1', layers[-1], [3, 3, conv_filter*8, conv_filter*8], stride = 1, wd = weight_decacy)
        relu7_1 = tf.nn.relu(conv7_1, name = 'relu7_1')

        conv7_2 = conv('conv7_2', relu7_1, [3, 3, conv_filter*8, conv_filter*8], stride = 1, wd = weight_decacy)
        relu7_2 = tf.nn.relu(conv7_2, name = 'relu7_2')

        conv7_3 = conv('conv7_3', relu7_2, [3, 3, conv_filter * 8, conv_filter * 8], stride=1, wd=weight_decacy)
        relu7_3 = tf.nn.relu(conv7_3, name='relu7_3')

        bn_7 = batch_norm('conv7_3norm', relu7_3, train = True)
        layers.append(bn_7)
        print("layer7: ",bn_7)

    with tf.variable_scope("layer8"):
        conv8_1 = deconv('conv8_1', layers[-1], [4, 4, conv_filter*8, conv_filter*4], stride = 2)

        # short cut: unet 구조를 만들기 위해 건너 뛰는 과정
        conv3_3_short = conv('conv3_3_short', bn_3, [3, 3, conv_filter*4, conv_filter*4], stride = 1, _bias = 1)

        conv8_1_comb = tf.add(conv8_1, conv3_3_short, name = 'conv8_1_comb')
        relu8_1 = tf.nn.relu(conv8_1_comb, name = 'relu8_1')

        conv8_2 = conv('conv8_2', relu8_1, [3, 3, conv_filter*4, conv_filter*4], stride = 1, wd = weight_decacy)
        relu8_2 = tf.nn.relu(conv8_2, name ='relu8_2')

        conv8_3 = conv('conv8_3', relu8_2, [3, 3, conv_filter*4, conv_filter*4], stride = 1, wd = weight_decacy)
        relu8_3 = tf.nn.relu(conv8_3, name ='relu8_3')

        bn_8 = batch_norm('conv8_3norm', relu8_3, train = True)
        layers.append(bn_8)
        print("layer8: ",bn_8)

    with tf.variable_scope("layer9"):
        conv9_1 = deconv('conv9_1', layers[-1], [4, 4, conv_filter*4, conv_filter*2], stride = 2, _bias = 1)

        # short cut
        conv2_2_short = conv('conv2_2_short', bn_2, [3, 3, conv_filter*2, conv_filter*2],stride = 1, _bias = 1)

        conv9_1_comb = tf.add(conv9_1, conv2_2_short, name = 'conv9_1_comb')
        relu9_1 = tf.nn.relu(conv9_1_comb, name='relu9_1')

        conv9_2 = conv('conv9_2', relu9_1, [3, 3, conv_filter*2, conv_filter*2], stride = 1, _bias = 1)
        relu9_2 = tf.nn.relu(conv9_2, name = 'relu9_2')

        bn_9 = batch_norm('conv9_2norm', relu9_2, train = True)
        layers.append(bn_9)
        print("layer9: ",bn_9)

    with tf.variable_scope("layer10"):
        conv10_1 = deconv('conv10_1', layers[-1], [4, 4, conv_filter*2, conv_filter*2], stride = 2, _bias = 1)

        # short cut
        conv1_2_short = conv('conv1_2_short', bn_1, [3, 3, conv_filter, conv_filter*2], stride = 1, _bias = 1)

        conv10_1_comb = tf.add(conv10_1, conv1_2_short, name = 'conv10_1_comb')
        relu10_1 = tf.nn.relu(conv10_1_comb, name = 'relu10_1')

        conv10_2 = conv('conv10_2', relu10_1, [3, 3, conv_filter*2, conv_filter*2], stride = 1, _bias = 1)
        relu10_2 = tf.nn.relu(conv10_2, name = 'relu10_2')
        print("layer10: ", relu10_2)

    pred_ab = tf.nn.tanh(conv('conv10_ab', relu10_2, [1, 1, conv_filter*2, 2],stride = 1),name = 'pred_ab')
    print("predicted ab tensor: ", pred_ab)

    return pred_ab