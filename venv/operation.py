import tensorflow as tf
import numpy as np

_weight_decacy = 0.01
stddev = 5e-2

def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer= initializer, dtype= tf.float32)
    return var

def variable_with_weight_decacy(name, shape, stddev, wd):
    var = _variable(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decacy = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection('losses', weight_decacy)
    return var

def conv(name, _input, kernel_size, stride = 1, dilation = 1, wd = _weight_decacy, _bias = 0, padding = 'SAME'):
    scope = name
    with tf.variable_scope(scope):
        kernel = variable_with_weight_decacy('weights', shape = kernel_size, stddev = stddev, wd = wd)

        if dilation == 1:
            conv = tf.nn.conv2d(_input, kernel, [1, stride, stride, 1], padding = padding)
        else:
            conv = tf.nn.atrous_conv2d(_input, kernel, dilation, padding=padding)

        if _bias == 1:
            biases = tf.get_variable('biases', kernel_size[3:],dtype=tf.float32,initializer= tf.constant_initializer(1.0))
        else:
            initializer = tf.zeros_initializer()
            # biases = tf.get_variable('biases', kernel_size[3:], tf.constant_initializer(0.0))
            biases = tf.get_variable('biases', kernel_size[3:], dtype= tf.float32,initializer= initializer)
        bias = tf.nn.bias_add(conv, biases)
        return bias

def conv_ds(name, _input, kernel_shape, stride = 2):
    scope = name
    with tf.variable_scope(scope):
        kernel = tf.get_variable('weights', shape= kernel_shape,
                                 initializer= tf.constant_initializer(1.), trainable= False)
        convl = tf.nn.depthwise_conv2d(_input, kernel, strides = [1, stride, stride, 1], padding = 'VALID')
        return convl

def deconv(name, _input, kernel_size, stride = 1, wd = _weight_decacy, _bias = 0):
    scope = name
    with tf.variable_scope(scope):
        batch_size, height, width, in_channel = [int(i) for i in _input.get_shape()]
        out_channel = kernel_size[3]
        kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
        output_shape = [batch_size, height * stride, width * stride, out_channel]
        kernel = variable_with_weight_decacy('weights', shape=kernel_size, stddev=stddev, wd=wd)
        deconv = tf.nn.conv2d_transpose(_input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
        if _bias == 1:
            biases = _variable('biases', (out_channel), tf.constant_initializer(1.0))
        else:
            biases = _variable('biases', (out_channel), tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(deconv, biases)

        return bias

def batch_norm(scope, x, train = True, reuse = False):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train,
                                        trainable=True, scope=scope)